import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as t

class CPTADDataset(Dataset):

    def __init__(self, label_file, data_path, nframes=16, transform=None):
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
        self.transform = transform ###

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        video_name = frame_list.iloc[idx, 0]
        frame_num = frame_list.iloc[idx, 1]
        class_label = frame_list.iloc[idx, 2]

        filename = '/notebooks/data/Datasets/CPTAD/Videos/' + video_name

        segment,_,_ = torchvision.io.read_video(filename,
                                               start_pts=(frame_num - self.dim[0])/30,
                                               end_pts=frame_num/30,
                                               pts_unit='sec',
                                               output_format="THWC")

        segment = segment.permute([3, 0, 1, 2]), # THWC -> CTHW

        if self.transform:
            segment = self.transform(segment)

        class_label = torch.tensor([int(class_label)])
        
        return seg_tensor, class_label