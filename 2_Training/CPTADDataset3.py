import numpy as np
import pandas as pd
import torch
import torchvision
import cv2
import itertools
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as t
import torchvision.transforms.functional as F
from random import randrange
import xml.etree.ElementTree as ET

class CPTADDataset3(Dataset):

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
        
    def __init__(self, video_name, video_folder, video_data_file, nframes=16, frame_transform=None, video_transform=None, 
                overlapthreshold=0.5):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the video.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.video_name = video_name
        self.tree = ET.parse(video_data_file)
        self.video_folder = video_folder
        self.nframes = nframes
        self.resolution = (112, 112) #https://arxiv.org/pdf/1711.11248
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self.overlapthreshold = overlapthreshold

    def __len__(self):
        return 128

    def __getitem__(self, idx):
        
        filename = '/notebooks/data/Datasets/CPTAD/Videos/' + self.video_name
        cap = cv2.VideoCapture(filename)
        
        root = self.tree.getroot()
        video = root.find(f"./video[@taskid='{436027}']") #find video with task_id
        height = int(video.find('height').text)
        width = int(video.find('width').text)
        
        numtracks = int(video.find('numtracks').text)
        tracks = video.find('tracks')
        maxoverlap = 0

        if idx % 2 == 0:
            # attempt to load crash example
            start = max(int(video.find('crashstart').text), 15)
            end = int(video.find('crashsettled').text)

            chosenframe = randrange(start, end)

            crash_frame = tracks[0].find(f"./frame[@frame='{chosenframe}']") # watch out for 0 indexing
            xtl = int(round(float(crash_frame.attrib.get('xtl'))))
            ytl = int(round(float(crash_frame.attrib.get('ytl'))))
            xbr = int(round(float(crash_frame.attrib.get('xbr'))))
            ybr = int(round(float(crash_frame.attrib.get('ybr'))))

            patchxtl = randrange(xtl, xbr) - 56 # 112/2
            patchytl = randrange(ytl, ybr) - 56

        else:
            # attempt to load no crash example
            start = 15
            end = int(video.find('crashsettled').text)

            chosenframe = randrange(start, end)

            patchxtl = randrange(0, width-112)
            patchytl = randrange(0, height-112)

        cap.set(cv2.CAP_PROP_POS_FRAMES, chosenframe - 15)
        frames = np.ndarray((16, height, width, 3))
        for i in range(16):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            frames[i] = frame

        tensor = torch.tensor(frames)
        tensor = torch.permute(tensor, (0, 3, 1, 2))
        tensor = F.crop(tensor, patchytl, patchxtl, 112, 112)
        tensor = F.normalize(tensor, mean=[0.43216, 0.394666, 0.37645],std=[0.22803, 0.22145, 0.216989])
        tensor = torch.permute(tensor, (1, 0, 2, 3))

        numtracks = int(video.find('numtracks').text)
        tracks = video.find('tracks')
        maxoverlap = 0
       
        for i in range(numtracks):
            bbox_frame = tracks[i].find(f"./frame[@frame='{chosenframe}']")
            if bbox_frame is not None:
                xtl = float(bbox_frame.attrib.get('xtl'))
                ytl = float(bbox_frame.attrib.get('ytl'))
                xbr = float(bbox_frame.attrib.get('xbr'))
                ybr = float(bbox_frame.attrib.get('ybr'))
                bboxA = [xtl, ytl, xbr, ybr]
                bboxB = [patchxtl, patchytl, patchxtl+111, patchytl+111]
                curroverlap = CPTADDataset3.overlap(bboxA, bboxB)
                if curroverlap > maxoverlap:
                    maxoverlap = curroverlap
        
        if maxoverlap >= self.overlapthreshold:
            class_label = 1
        else:
            class_label = 0
            
        label = torch.tensor(int(class_label))
        
        return tensor, label   
