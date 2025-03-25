import math
import os
import time
import random

import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm
import madgrad

from datasets.echonet_dynamic import EchoNet

def set_seed(s):
   torch.manual_seed(s)
   torch.cuda.manual_seed_all(s)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   np.random.seed(s)
   random.seed(s)
   os.environ['PYTHONHASHSEED'] = str(s)

def get_optimizer(model, args):
    if args.optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer_name == "adamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_name == "madgrad":
        optimizer = madgrad.MADGRAD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    return optimizer

def get_lr_scheduler(optimizer, args):
    if args.lr_scheduler == "step":
        if args.lr_step_period is None:
            lr_step_period = math.inf
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_period)

    return lr_scheduler


def get_mean_and_sd(dataset: torch.utils.data.Dataset,
                     num_samples: int = 128,
                     batch_size: int = 8,
                     num_workers: int = 4):


    if num_samples is not None and len(dataset) > num_samples:
        dataset = torch.utils.data.Subset(dataset,
                np.random.choice(len(dataset), num_samples, replace=False))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    samples, sum1, sum2 = 0, 0., 0.
    for (video, *_) in tqdm.tqdm(dataloader):
        video = video.transpose(0, 1).contiguous().view(3, -1)
        sum1 += torch.sum(video, dim=1).numpy()
        sum2 += torch.sum(video ** 2, dim=1).numpy()
        samples += video.shape[1]

    mean = (sum1 / samples).astype(np.float32)
    sd = (np.sqrt(sum2 / samples - mean ** 2)).astype(np.float32)

    return mean, sd


def bootstrap_metric(arg1, arg2, fun, num_samples=10000):
    results = []
    arg1, arg2 = np.array(arg1), np.array(arg2)

    for _ in range(num_samples):
        index = np.random.choice(len(arg1), len(arg1))
        results.append(fun(arg1[index], arg2[index]))

    results = sorted(results)
    percentile_05 = results[round(0.05 * len(results))]
    percentile_95 = results[round(0.95 * len(results))]

    return fun(arg1, arg2), percentile_05, percentile_95

def run_epoch(model, dataloader, train, optimizer, device):
    model.train(train)

    total_loss, videos = 0, 0
    total_reg = 0
    total_seg = 0
    y, yhat = [], []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as progressbar:
            for (video, ef, mask) in dataloader:

                mask = mask.cuda()
                bz, nc, nf, h, w = mask.shape
                y.append(ef.numpy())
                video = video.to(device)
                ef = ef.to(device) # batch

                average = (len(video.shape) == 6)
                if average:
                    batch_size, num_clips, c, f, h, w = X.shape
                    video = video.view(-1, c, f, h, w)

                outputs, pred_mask = model(video)

                if average:
                    outputs = outputs.view(batch_size, num_clips, -1).mean(1)

                yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                loss_reg = torch.nn.functional.mse_loss(outputs.view(-1), ef)
                mask = mask.reshape(bz*nf,nc,h,w)

                loss_seg = structure_loss(pred_mask, mask)
                loss = loss_reg + loss_seg


                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * video.size(0)
                total_reg += loss_reg.item() * video.size(0)
                total_seg += loss_seg.item() * video.size(0)

                videos += video.size(0)

                progressbar.set_postfix_str("{:.2f} ({:.2f})".format(total_loss / videos, loss.item()))
                progressbar.update()

    yhat = np.concatenate(yhat)
    y = np.concatenate(y)
    total_loss = total_loss / videos
    total_reg = total_reg / videos
    total_seg = total_seg / videos
    print("loss && reg && seg",total_loss, total_reg, total_seg)
    return total_loss, total_reg, total_seg, yhat, y


def run_train(output, device, model, optimizer, lr_scheduler, bestLoss, bestR2, epoch_resume, wandb, f, args):
    train_ds = EchoNet(
            root=args.data_dir,
            split="train",
            mean=args.mean,
            std=args.std,
            frames=args.frames,
            frequency=args.frequency,
            pad=12,
            segin_dir = "train_infer_mask"
    )
    val_ds = EchoNet(
            root=args.data_dir,
            split="val",
            mean=args.mean,
            std=args.std,
            frames=args.frames,
            frequency=args.frequency,
            segin_dir="val_infer_mask"
    )

    for epoch in range(epoch_resume, args.epochs):
        print("Epoch #{}".format(epoch), flush=True)
        for phase in ['train', 'val']:
            start_time = time.time()
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)

            dataset = train_ds if phase == "train" else val_ds
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
                pin_memory=(device.type == "cuda"),
                drop_last=(phase == "train")
            )

            loss, loss_reg, loss_seg, yhat, y = run_epoch(model, dataloader, phase == "train", optimizer, device)
            f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                            phase,
                                                            loss,
                                                            sklearn.metrics.r2_score(y, yhat),
                                                            time.time() - start_time,
                                                            y.size,
                                                            sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                            sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                            args.batch_size,
                                                            loss_reg,
                                                            loss_seg))
            if phase == "train":
                wandb.log({"epoch": epoch, "train loss": loss, "train r2": sklearn.metrics.r2_score(y, yhat)})
            else:
                wandb.log({"epoch": epoch, "val loss": loss, "val r2": sklearn.metrics.r2_score(y, yhat)})


            f.flush()
        lr_scheduler.step()

        # Save checkpoint
        save = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'frequency': args.frequency,
            'frames': args.frames,
            'best_loss': bestLoss,
            'loss': loss,
            'bestR2': bestR2,
            'r2': sklearn.metrics.r2_score(y, yhat),
            'opt_dict': optimizer.state_dict(),
            'scheduler_dict': lr_scheduler.state_dict(),
        }
        torch.save(save, os.path.join(output, "checkpoint.pt"))
        if loss_reg < bestLoss:
            torch.save(save, os.path.join(output, "best.pt"))
            bestLoss = loss_reg
        r2 = sklearn.metrics.r2_score(y, yhat)
        if r2 > bestR2:
            torch.save(save, os.path.join(output, "best_r2.pt"))
            bestR2 = r2

def run_test(output, device, model, wandb, f, args):
    if args.epochs != 0:
        checkpoint = torch.load(os.path.join(output, "best.pt"))
        model.load_state_dict(checkpoint['state_dict'])
        print(os.path.join(output, "best.pt"))
        f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
        f.flush()

    for split in ["test","val"]:
        set_seed(0)
        dataset = EchoNet(
                root=args.data_dir,
                split=split,
                mean=args.mean,
                std=args.std,
                frames=args.frames,
                frequency=args.frequency,
                segin_dir="directory of seg mask/{}_infer_mask".format(split)
        )
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
                pin_memory=(device.type == "cuda")
        )
        loss, loss_reg, loss_seg, yhat, y = run_epoch(model, dataloader, False, None, device)

        r2  = bootstrap_metric(y, yhat, sklearn.metrics.r2_score)
        mae = bootstrap_metric(y, yhat, sklearn.metrics.mean_absolute_error)
        rmse = tuple(map(math.sqrt, bootstrap_metric(y, yhat, sklearn.metrics.mean_squared_error)))

        print("R2: ", sklearn.metrics.r2_score(y, yhat))

        f.write("{} R2:   {:.5f} ({:.5f} - {:.5f})\n".format(split, *r2))
        f.write("{} MAE:  {:.5f} ({:.5f} - {:.5f})\n".format(split, *mae))
        f.write("{} RMSE: {:.5f} ({:.5f} - {:.5f})\n".format(split, *rmse))
        f.flush()

        if split == "val":
            wandb.log({split + " test loss": loss})
        else:
            wandb.log({split + " loss": loss})

        wandb.log({split + " r2": r2[0]})
        wandb.log({split + " mae": mae[0]})
        wandb.log({split + " rmse": rmse[0]})
