"""EchoNet Dataset."""

import os
import collections

import matplotlib.pyplot as plt
import pandas
import cv2

import numpy as np
import skimage.draw
import torchvision
from scipy.special import expit
class EchoNet(torchvision.datasets.VisionDataset):
    def __init__(self, root=None,
                 split="train",
                 mean=0.,
                 std=1.,
                 frames=16,
                 frequency=2,
                 max_frames=250,
                 pad=None,
                 segin_dir = None):

        assert(root is not None)

        super().__init__(root)

        self.split = split
        self.mean = mean
        self.std = std
        self.frames = frames
        self.max_frames = max_frames
        self.frequency = frequency
        self.pad = pad
        self.segin_dir = segin_dir

        self.vnames, self.outcome = [], []
        self.read_filelist()

        self.frames_list = collections.defaultdict(list)
        self.trace = collections.defaultdict(_defaultdict_of_lists)

        self.read_volumetracings()

        self.filter_videos()

        print("{} dataset size: {}".format(split, len(self.vnames)))

    def read_filelist(self):
        with open(os.path.join(self.root, "FileList.csv")) as f:
            self.file_header = f.readline().strip().split(",")
            filename_index = self.file_header.index("FileName")
            split_index = self.file_header.index("Split")

            for line in f:
                line_split = line.strip().split(',')

                filename = os.path.splitext(line_split[filename_index])[0] + ".avi"
                file_split = line_split[split_index].lower()

                if self.split in ["all", file_split] and os.path.exists(os.path.join(self.root, "Videos", filename)):
                    self.vnames.append(filename)
                    self.outcome.append(line_split)

        self.check_missing_videos()


    def check_missing_videos(self):
        missing_videos = set(self.vnames) - set(os.listdir(os.path.join(self.root, "Videos")))
        if len(missing_videos) != 0:
            print("{} videos are missing in {}:".format(len(missing_videos), os.path.join(self.root, "Videos")))
            for f in sorted(missing):
                print("\t", f)
            raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing_videos)[0]))

    def read_volumetracings(self):
        with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
            header = f.readline().strip().split(",")
            assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

            for line in f:
                filename, x1, y1, x2, y2, frame = line.strip().split(',')
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                frame = int(frame)
                if frame not in self.trace[filename]:
                    self.frames_list[filename].append(frame)
                self.trace[filename][frame].append((x1, y1, x2, y2))

        for filename in self.frames_list:
            for frame in self.frames_list[filename]:
                 self.trace[filename][frame] = np.array(self.trace[filename][frame])


    def filter_videos(self):
        min_frames = 2
        videos_to_keep = [len(self.frames_list[f]) >= min_frames for f in self.vnames]
        self.vnames = [f for (f, k) in zip(self.vnames, videos_to_keep) if k]
        self.outcome = [f for (f, k) in zip(self.outcome, videos_to_keep) if k]

    def __getitem__(self, index):
        video = os.path.join(self.root, "Videos", self.vnames[index])

        video = self.load_video(video).astype(np.float32)
        video = self.normalize_video(video)

        if self.segin_dir:
            video, segmasks = self.sample_video(video, index)
            if self.pad is not None:
                video, segmasks = self.pad_video(video, segmasks) #
        else:
            video = self.sample_video(video, index)
            if self.pad is not None:
                video = self.pad_video(video)

        ef = np.float32(self.outcome[index][self.file_header.index("EF")])

        if self.segin_dir:
            return video, ef, segmasks
        else:
            return video, ef

    def __len__(self):
        return len(self.vnames)

    def normalize_video(self, video):
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        return video

    def load_mask(self, index, video):
        # load ed es & mask
        key = self.vnames[index]
        target = []
        target.append(video[:, self.frames_list[key][-1], :, :]) # largeframe
        target.append(video[:, self.frames_list[key][0], :, :]) # smallframe
        for t in ["LargeTrace", "SmallTrace"]:
            if t == "LargeTrace":
                t = self.trace[key][self.frames_list[key][-1]]
            else:
                t = self.trace[key][self.frames_list[key][0]]
            x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
            x = np.concatenate((x1[1:], np.flip(x2[1:])))
            y = np.concatenate((y1[1:], np.flip(y2[1:])))

            r, c = skimage.draw.polygon(np.rint(y).astype(np.int32), np.rint(x).astype(np.int32),
                                        (video.shape[2], video.shape[3]))
            mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
            mask[r, c] = 1
            target.append(mask)
        return target

    def sample_video(self, video, index):
        c, f, h, w = video.shape
        frames = self.frames
        frames = min(frames, self.max_frames)

        if f < frames * self.frequency:
            video = np.concatenate((video, np.zeros((c, frames * self.frequency - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  # pylint: disable=E0633

        start = np.random.choice(f - (frames - 1) * self.frequency, 1)

        video = tuple(video[:, s + self.frequency * np.arange(frames), :, :] for s in start)[0]

        # todo load numpy mask


        if self.segin_dir:
            seg_infer_path = os.path.join(self.segin_dir, self.vnames[index].replace(".avi", ".npy"))
            seg_infer_logits = np.load(seg_infer_path)
            digiseg = True
            if digiseg:
                seg_infer_probs = (seg_infer_logits > 0).astype(np.float32)

            else:
                seg_infer_probs = expit(seg_infer_logits)
            seg_infer_prob_norm = seg_infer_probs

            #### check if need to append
            fs = seg_infer_prob_norm.shape[0]

            if fs < frames * self.frequency:
                seg_infer_prob_norm = np.concatenate(
                    (seg_infer_prob_norm, np.ones((frames * self.frequency - fs, h, w), video[0].dtype) * -1), axis=0)

            seg_infer_prob_norm = np.expand_dims(seg_infer_prob_norm, axis=0)

            seg_infer_prob_norm_samp = tuple(
            seg_infer_prob_norm[:, s + self.frequency * np.arange(frames), :, :] for s in start)[0]

            return video, seg_infer_prob_norm_samp
        else:
            return video


    def pad_video(self, video, segmasks):
        if self.pad is None:
            return video, segmasks

        c, l, h, w = video.shape
        tvideo = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
        tvideo[:, :, self.pad:-self.pad, self.pad:-self.pad] = video  # pylint: disable=E1130
        i, j = np.random.randint(0, 2 * self.pad, 2)

        tmasks = np.zeros((1, l, h + 2 * self.pad, w + 2 * self.pad), dtype=segmasks.dtype)
        tmasks[:, :, self.pad:-self.pad, self.pad:-self.pad] = segmasks  # pylint: disable=E1130

        return tvideo[:, :, i:(i + h), j:(j + w)], tmasks[:, :, i:(i + h), j:(j + w)]

    def load_video(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        capture = cv2.VideoCapture(path)

        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video = np.zeros((count, height, width, 3), np.uint8)

        for i in range(count):
            out, frame = capture.read()
            if not out:
                raise ValueError("Problem when reading frame #{} of {}.".format(i, filename))

            video[i, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return video.transpose((3, 0, 1, 2))

def _defaultdict_of_lists():
    return collections.defaultdict(list)
