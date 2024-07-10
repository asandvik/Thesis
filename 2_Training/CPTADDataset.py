import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import itertools
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as t
from random import randrange

class CPTADDataset(Dataset):
    
    def intervals(self, idx):
        name = self.data.iloc[idx, 0]
        interval_start = self.data.iloc[idx, 1]
        interval_end = self.data.iloc[idx, 2]
        label = self.data.iloc[idx, 3]

        last_frame = randrange(interval_start, interval_end)

        filename = '/notebooks/data/Datasets/CPTAD/Videos/' + name

        video = torchvision.io.VideoReader(filename, "video")
        metadata = video.get_metadata()
        fps = metadata["video"]['fps'][0]
        frames = []

        start_s = (last_frame - (self.nframes-1))/fps
        for frame in itertools.islice(video.seek(start_s), self.nframes):
            frames.append(frame['data'])
        segment = torch.stack(frames, 0)
        
        if self.video_transform:
               segment = self.video_transform(segment)
            
        label = torch.tensor(int(label))
        return segment, label

    

    def windows(self, idx):
        video_name = self.data.iloc[idx, 0]
        frame_num = self.data.iloc[idx, 1]
        class_label = self.data.iloc[idx, 2]

        filename = '/notebooks/data/Datasets/CPTAD/Videos/' + video_name

        video = torchvision.io.VideoReader(filename, "video")
        metadata = video.get_metadata()
        fps = metadata["video"]['fps'][0]
        frames = []

        start = (frame_num - self.nframes)/fps
        for frame in itertools.islice(video.seek(start), self.nframes):
            frames.append(frame['data'])
            #current_pts = frame['pts']

        segment = torch.stack(frames, 0)
        if self.video_transform:
               segment = self.video_transform(segment)

        # segment = segment.permute([3, 0, 1, 2]), # THWC -> CTHW

        label = torch.tensor(int(class_label))

        # print("{}. {}".format(idx, label.size()))
        
        return segment, label        

    def __init__(self, label_file, data_path, nframes=8, frame_transform=None, video_transform=None, sample_tech='regions'):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the video.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(label_file,header=None)
        self.data_path = data_path
        self.nframes = nframes
        self.resolution = (112, 112) #https://arxiv.org/pdf/1711.11248
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        if sample_tech == 'intervals':
            self.sample = self.intervals
        elif sample_tech == 'windows':
            self.sample = self.windows
        else:
            error("Invalid sample_tech")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.sample(idx)
