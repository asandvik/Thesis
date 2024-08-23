import numpy as np
import pandas as pd
import torch
import torchvision
import itertools
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as t
from random import randrange
import xml.etree.ElementTree as ET

class CPTADDataset2(Dataset):

    def overlap(boxA, boxB):
        areaA = float((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = float((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        intAB = [max(boxA[0], boxB[0]), # xtl
                 max(boxA[1], boxB[1]), # ytl
                 min(boxA[2], boxB[2]), # xbr
                 min(boxA[3], boxB[3])] # ybr

        if intAB[0] > intAB[2] or intAB[1] > intAB[3]:
            return 0

        areaAB = float((intAB[2] - intAB[0]) * (intAB[3] - intAB[1]))

        return areaAB / min(areaA, areaB)
        
    def __init__(self, region_file, data_path, video_data_file, nframes=16, frame_transform=None, video_transform=None, 
                 sample_tech='regions', overlapthreshold=0.5):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the video.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.regions = pd.read_csv(region_file,header=None)
        self.tree = ET.parse(video_data_file)
        self.data_path = data_path
        self.nframes = nframes
        self.resolution = (112, 112) #https://arxiv.org/pdf/1711.11248
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self.overlapthreshold = overlapthreshold
        if sample_tech == 'intervals':
            self.sample = self.intervals
        elif sample_tech == 'windows':
            self.sample = self.windows
        elif sample_tech == 'regions':
            self.sample = self.regions
        else:
            raise Exception("Invalid sample_tech")

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        video_id = self.regions.iloc[idx, 0]
        video_name = self.regions.iloc[idx, 1]
        tlow = self.regions.iloc[idx, 2]
        thigh = self.regions.iloc[idx, 3]
        hlow = self.regions.iloc[idx, 4]
        hhigh = self.regions.iloc[idx, 5]
        wlow = self.regions.iloc[idx, 6]
        whigh = self.regions.iloc[idx, 7]

        t0 = randrange(tlow, thigh+1)
        h0 = randrange(hlow, hhigh+1)
        w0 = randrange(wlow, whigh+1)
        
        filename = '/notebooks/data/Datasets/CPTAD/Videos/' + video_name
        reader = torchvision.io.VideoReader(filename, "video")
        metadata = reader.get_metadata()
        fps = metadata["video"]['fps'][0]
        frames = []

        start_s = t0/fps
        reader.seek(start_s)
        for _ in range(self.nframes):
            frame = next(reader)
            frames.append(t.functional.crop(frame['data'], h0, w0, 112, 112))
                
        segment = torch.stack(frames, 0)
        if self.video_transform:
               segment = self.video_transform(segment)

        tlast = t0 + self.nframes - 1

        root = self.tree.getroot()
        video = root.find(f"./video[@taskid='{video_id}']") #find video with task_id
        numtracks = int(video.find('numtracks').text)
        tracks = video.find('tracks')
        maxoverlap = 0
       
        for i in range(numtracks):
            bbox_frame = tracks[i].find(f"./frame[@frame='{tlast}']")
            if bbox_frame is not None:
                xtl = float(bbox_frame.attrib.get('xtl'))
                ytl = float(bbox_frame.attrib.get('ytl'))
                xbr = float(bbox_frame.attrib.get('xbr'))
                ybr = float(bbox_frame.attrib.get('ybr'))
                bboxA = [xtl, ytl, xbr, ybr]
                bboxB = [w0, h0, w0+111, h0+111]
                curroverlap = CPTADDataset2.overlap(bboxA, bboxB)
                if curroverlap > maxoverlap:
                    maxoverlap = curroverlap
        
        if maxoverlap >= self.overlapthreshold:
            class_label = 1
        else:
            class_label = 0
            
        label = torch.tensor(int(class_label))

        # print(segment.size(),video_id, tlow, thigh, hlow, hhigh, wlow, whigh)
        
        return segment, label   
