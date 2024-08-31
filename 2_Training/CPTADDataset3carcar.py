import os
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

class CPTADDataset3carcar(Dataset):

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
        
    def __init__(self, video_list, video_folder, norm_video_folder, video_data_file, nframes=16, frame_transform=None, video_transform=None, 
                overlapthreshold=0.5, nspi = 4):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the video.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.video_list = pd.read_csv(video_list, header=None)
        self.tree = ET.parse(video_data_file)
        self.video_folder = video_folder
        self.norm_video_folder = norm_video_folder
        self.norm_video_list = os.listdir(norm_video_folder)
        self.norm_idx = 0
        self.nframes = nframes
        self.resolution = (112, 112) #https://arxiv.org/pdf/1711.11248
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self.overlapthreshold = overlapthreshold
        self.nspi = nspi # number samples per impact. must be multiple of 4
        self.rng = np.random.default_rng()
        self.mvmnt = 5 # same as NFRAMES_MD in applyModel.py
        # self.foc = foc # frequency of crash vs non crash. 2 = one in ever two. 3 = one in every three

    def __len__(self):
        return len(self.video_list) * self.nspi

    def __getitem__(self, idx):

        # 0 - crash
        # 1 - no crash
        # 2 - crash
        # 3 - normal
        source = idx % 4 

        impact = int(idx / self.nspi)

        # get video path

        # if no crash/normal, get reference frame

        if source == 3: # if normal example
            video_name = self.norm_video_list[self.norm_idx]
            self.norm_idx = (self.norm_idx + 1) % len(self.norm_video_list) # set to next index
            
            filename = self.norm_video_folder + video_name

        else: # if crash/no crash example
        
            video_id = self.video_list.iloc[impact, 0]
            video_name = self.video_list.iloc[impact, 1]
            track_idx = self.video_list.iloc[impact, 2]
            
            filename = '/notebooks/data/Datasets/CPTAD/Videos/' + video_name
            
            root = self.tree.getroot()
            video = root.find(f"./video[@taskid='{video_id}']") #find video with task_id
            # height = int(video.find('height').text)
            # width = int(video.find('width').text)
            # length = int(video.find('length').text)
            
            numtracks = int(video.find('numtracks').text)
            tracks = video.find('tracks')
            maxoverlap = 0

        
        cap = cv2.VideoCapture(filename)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if length == 0:
            raise ValueError(f"{video_name} has length 0")

        # get chosen frame
        if source == 0 or source == 2: # attempt to load crash example
            
            start = max([int(tracks[track_idx].attrib.get('start')), self.mvmnt-1, self.nframes-1])
            settled = int(tracks[track_idx].attrib.get('settled'))
            end = int(tracks[track_idx].attrib.get('end'))

            if settled > start:
                sigma = (settled - start) / 3
            elif end > start:
                sigma = (end - start) / 3
            elif length > start:
                sigma = (length - start) / 3
            else:
                raise ValueError(f"Video {video_id} is too short")
                
            offset = abs(self.rng.normal(loc=0.0,scale=sigma))
            while offset > sigma*3:
                offset -= sigma*3
            chosenframe = int(round(start + offset))
            # chosenframe = randrange(start, min(settled+self.nframes, end))

            # while True: # handle possible missing frames from outside characteristic
            crash_frame = tracks[track_idx].find(f"./frame[@frame='{chosenframe}']") # watch out for 0 indexing
            if crash_frame is None:  # grab random frame in video
                print("Missing Frame", idx, video_name, track_idx, chosenframe, start, settled, end)
                chosenframe = randrange(start, length)
                crash_frame = tracks[track_idx].find(f"./frame[@frame='{chosenframe}']")

            if crash_frame is None:
                print(f"{video_id} Gave up")
                source = 1 # give up
            else:
                xtl = int(round(float(crash_frame.attrib.get('xtl'))))
                ytl = int(round(float(crash_frame.attrib.get('ytl'))))
                xbr = int(round(float(crash_frame.attrib.get('xbr'))))
                ybr = int(round(float(crash_frame.attrib.get('ybr'))))

                patchxtl = randrange(xtl, xbr) - 56 # 112/2
                patchytl = randrange(ytl, ybr) - 56

        elif source == 1: # attempt to load no crash example

            start = max([self.nframes-1, self.mvmnt-1])
            settled = int(tracks[track_idx].attrib.get('settled'))
            end = int(tracks[track_idx].attrib.get('end'))

            # bc of task_id 436793. crash ends before frame 31 (start)
            upperbound = min(settled+self.nframes, end) if min(settled+self.nframes, end) > start else length

            chosenframe = randrange(start, upperbound)
        
        else: # normal example
            chosenframe = randrange(max([self.mvmnt-1,self.nframes-1]), length)

        # load frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, chosenframe - (self.nframes-1))
        frames = np.ndarray((self.nframes, height, width, 3))
        for i in range(self.nframes):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            frames[i] = frame

        # if non crash or normal sample, get location of patch
        if source == 1 or source == 3:
            # get reference frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, chosenframe - (self.mvmnt-1))
            ret, ref = cap.read()

            # movement
            graynow = cv2.cvtColor(frames[-1].astype(np.float32), cv2.COLOR_RGB2GRAY)
            graynow = cv2.GaussianBlur(graynow, (21, 21), 0)
            grayref = cv2.cvtColor(ref.astype(np.float32), cv2.COLOR_BGR2GRAY)
            grayref = cv2.GaussianBlur(grayref, (21, 21), 0)
            delta = cv2.absdiff(graynow, grayref)
            thresh = cv2.threshold(delta, 35, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=1)
            y_idx, x_idx = np.where(thresh==255) # lists of movement pixels
            if len(y_idx) == 0:
                patchxtl = randrange(0, width-112)
                patchytl = randrange(0, height-112)
            else:
                rand_idx = np.random.choice(len(x_idx))
                # print(rand_idx, len(y_idx), len(x_idx))
                patchytl = max(0, y_idx[rand_idx] - 56)
                patchxtl = max(0, x_idx[rand_idx] - 56)

        # need patchxtl and patchytl

        tensor = torch.tensor(frames)
        tensor = torch.permute(tensor, (0, 3, 1, 2))
        tensor = F.crop(tensor, patchytl, patchxtl, 112, 112)
        tensor = F.normalize(tensor, mean=[0.43216, 0.394666, 0.37645],std=[0.22803, 0.22145, 0.216989])
        tensor = torch.permute(tensor, (1, 0, 2, 3)) #TCHW
                
        if self.video_transform:
            tensor = self.video_transform(tensor)

        if source == 3:
            class_label = 0
        else:
            numtracks = int(video.find('numtracks').text)
            tracks = video.find('tracks')
            maxoverlap = 0
        
            for i in range(numtracks):
                bbox_frame = tracks[i].find(f"./frame[@frame='{chosenframe}']")
                el1 = tracks[i].attrib.get("Element1")
                el2 = tracks[i].attrib.get("Element2")
                if bbox_frame is not None and el1 == "Car" and el2 == "Car":
                    xtl = float(bbox_frame.attrib.get('xtl'))
                    ytl = float(bbox_frame.attrib.get('ytl'))
                    xbr = float(bbox_frame.attrib.get('xbr'))
                    ybr = float(bbox_frame.attrib.get('ybr'))
                    bboxA = [xtl, ytl, xbr, ybr]
                    bboxB = [patchxtl, patchytl, patchxtl+111, patchytl+111]
                    curroverlap = CPTADDataset3carcar.overlap(bboxA, bboxB)
                    if curroverlap > maxoverlap:
                        maxoverlap = curroverlap

            if maxoverlap >= self.overlapthreshold:
                class_label = 1
            else:
                class_label = 0
            
        label = torch.tensor(int(class_label))

        
        return tensor, label   
