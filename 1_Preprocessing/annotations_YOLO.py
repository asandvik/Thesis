import xml.etree.ElementTree as ET
import matplotlib.pylab as plt
import numpy as np
import random
import csv
import os
import cv2
from math import floor
 
VIDEO_DIR = '/notebooks/data/Datasets/CPTAD/Videos/'
YOLO_DIR = '/notebooks/Thesis/data_yolo/'

tree = ET.parse('/notebooks/Thesis/1_Preprocessing/annotations_reformatted.xml')
videos = tree.getroot()

# List of suitable videos to use
id_list = []
for video in videos.iter('video'):
    status = video.find('status').text
    numtracks = int(video.find('numtracks').text)
    if status == 'accepted':
        tracks = video.find('tracks')
        for track in tracks:
            el1 = track.attrib.get('Element1')
            el2 = track.attrib.get('Element2')
            if el1 == 'Car' and el2 == 'Car': # if video contains a car/car crash at all
                id_list.append(video.get('taskid')) 
                break


random.Random(40).shuffle(id_list)

i = 0
num_vids = len(id_list)

def background_frames(settype, taskid):
    video = videos.find(f".//video[@taskid='{taskid}']")
    name = video.find('name').text
    crashstart = int(video.find('crashstart').text)
    currframe = max(0, crashstart-75)
    cap = cv2.VideoCapture(VIDEO_DIR + name)
    cap.set(cv2.CAP_PROP_POS_FRAMES, currframe)

    while currframe < max(0, crashstart-15): # don't give it background samples too close to crash
        currframe = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if frame is None:
            break
    
        prefix = f"{name[:-4]}_{currframe}_background"
        cv2.imwrite(YOLO_DIR+'images/'+settype+'/'+prefix+'.jpg', frame)

def frames_and_labels(settype, taskid):
    video = videos.find(f".//video[@taskid='{taskid}']")
    name = video.find('name').text
    tracks = video.find('tracks')
    
    # get first frame with car/car impact and last keypoints
    currframe = None
    settled = 0
    end = 0
    for track in tracks:
        el1 = track.attrib.get('Element1')
        el2 = track.attrib.get('Element2')
        if el1 == 'Car' and el2 == 'Car':
            if currframe is None:
                currframe = int(track.attrib.get('start'))
            settled = max(settled, int(track.attrib.get('settled')))
            end = max(settled, int(track.attrib.get('end')))

    if currframe is None:
        raise ValueError('currframe is not set')
    
    cutoff = min([end, settled + 30, currframe + 200])
            
    cap = cv2.VideoCapture(VIDEO_DIR + name)
    cap.set(cv2.CAP_PROP_POS_FRAMES, currframe)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    startframe = currframe
    while currframe < cutoff:
        currframe = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if frame is None:
            break

        # get normalized car/car bounding boxes of frame
        boxes = []
        for track in tracks:
            el1 = track.attrib.get('Element1')
            el2 = track.attrib.get('Element2')
            if el1 == 'Car' and el2 == 'Car':
                framedata = track.find(f"./frame[@frame='{currframe}']")
                if framedata is not None:
                    occluded = int(framedata.attrib.get('occluded'))
                    outside = int(framedata.attrib.get('outside'))

                    if outside != 0 or occluded != 0:
                        continue

                    xtl = float(framedata.attrib.get('xtl'))
                    ytl = float(framedata.attrib.get('ytl'))
                    xbr = float(framedata.attrib.get('xbr'))
                    ybr = float(framedata.attrib.get('ybr'))

                    x_center = ((xbr + xtl) / 2) / width
                    y_center = ((ybr + ytl) / 2) / height
                    w = (xbr - xtl) / width
                    h = (ybr - ytl) / height

                    boxes.append(f"0 {x_center} {y_center} {w} {h}\n")

        if len(boxes) > 0:
            prefix = f"{name[:-4]}_{currframe}"

            labelfile = open(YOLO_DIR+'labels/'+settype+'/'+prefix+'.txt', 'w', newline='')
            labelfile.writelines(boxes)
            labelfile.close()

            # cv2.imwrite(YOLO_DIR+'images/'+settype+'/'+prefix+'.jpg', frame)

    cap.release()

# testfile = open(YOLO_DIR + 'test_videos.txt', 'w', newline='')

# def add_to_test(taskid):
#     video = videos.find(f".//video[@taskid='{taskid}']")
#     name = video.find('name').text
#     testfile.write(f"{taskid} {name}\n")

    
for i in range(0, num_vids):
    if (i % 10 == 0):
        # frames_and_labels('val', id_list[i])
        background_frames('val', id_list[i])
    elif (i % 10 == 1):
        # add_to_test(id_list[i])
        pass
    else:
        # frames_and_labels('train', id_list[i])
        background_frames('train', id_list[i])

# testfile.close()