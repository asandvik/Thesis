"""
This script 
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import xml.etree.ElementTree as ET
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.widgets import Button, TextBox, Slider
import matplotlib.gridspec as gridspec
import cv2
import time

MIN_CONF = 0.25

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

    return areaAB

def iou(boxA, boxB):
    intersection = overlap(boxA, boxB)

    if intersection == 0:
        return 0

    areaA = float((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = float((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    union = areaA + areaB - intersection

    return intersection / union


def getAnnoBox(frame):
    
    xtl = float(frame.attrib.get('xtl'))
    ytl = float(frame.attrib.get('ytl'))
    xbr = float(frame.attrib.get('xbr'))
    ybr = float(frame.attrib.get('ybr'))

    return xtl, ytl, xbr, ybr

def getResBox(row):
    xtl = round(row['x'] - row['w']/2)
    ytl = round(row['y'] - row['h']/2)
    xbr = round(row['x'] + row['w']/2)
    ybr = round(row['y'] + row['h']/2)

    return xtl, ytl, xbr, ybr

def calcPreCrash(track):

    impactframe = track[0]
    futureframe = track[1]

    xtl0, ytl0, xbr0, ybr0 = getAnnoBox(impactframe)
    xtl1, ytl1, xbr1, ybr1 = getAnnoBox(futureframe)

    dxtl = xtl1 - xtl0
    dytl = ytl1 - ytl0
    dxbr = xbr1 - xbr0
    dybr = ybr1 - ybr0

    frame = int(impactframe.attrib.get('frame'))
    xtl = xtl0
    ytl = ytl0
    xbr = xbr0
    ybr = ybr0
    data = []
    for i in range(1,9): # TODO: ensure values stay within frame of video
        xtl -= dxtl
        ytl -= dytl
        xbr -= dxbr
        ybr -= dybr
        data.append([frame-i, xtl, ytl, xbr, ybr])

    return np.array(data)

def loadNormalResults(i):
    global norm_video_list, cap, yolo_boxes2, yolo_boxes4, tracks, preCrashBoxes, axLeft, length, video_name

    video_name = norm_video_list[i]

    video_path = f"/notebooks/data/Datasets/CPTAD/Videos_Normal/Test/{video_name}"
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    videoSlider.valmax = length-1
    videoSlider.ax.set_xlim(videoSlider.valmin, videoSlider.valmax)

    axLeft.set_title(video_name)
    axLeft.set_xlim(0, length)
    axLeft.set_ylim(0, width)
    axLeft.set_zlim(0, height)
    axLeft.invert_zaxis()
    axLeft.invert_xaxis()
    axLeft.set_xlabel('frame')
    axLeft.set_ylabel('width')
    axLeft.set_zlabel('height')
    axLeft.view_init(elev=25, azim=45)

    yolo_boxes2 = pd.read_csv(f"/notebooks/Thesis/results/yolo_train2_norm/{video_name[:-4]}_{i}.csv")
    yolo_boxes4 = pd.read_csv(f"/notebooks/Thesis/results/yolo_train4_norm/{video_name[:-4]}_{i}.csv")

    preCrashBoxes = []
    tracks = ET.Element("Empty")
    
def loadVideoResults():
    global videos, video_name, video_id, axLeft, yolo_boxes2, yolo_boxes4, tracks
    global axRight, length, cap, videoSlider, currFrame, height
    global preCrashBoxes
    video = videos.find(f".//video[@taskid='{video_id}']")
    length = int(video.find('length').text)
    width = int(video.find('width').text)
    height = int(video.find('height').text)
    tracks = video.find('tracks')

    plt.suptitle(video_name)

    axLeft.set_xlim(0, length)
    axLeft.set_ylim(0, width)
    axLeft.set_zlim(0, height)
    axLeft.invert_zaxis()
    axLeft.invert_xaxis()
    axLeft.set_xlabel('frame')
    axLeft.set_ylabel('width')
    axLeft.set_zlabel('height')
    axLeft.view_init(elev=25, azim=45)

    yolo_boxes2 = pd.read_csv(f"/notebooks/Thesis/results/yolo_train2/{video_name[:-4]}_{video_id}.csv")
    yolo_boxes4 = pd.read_csv(f"/notebooks/Thesis/results/yolo_train4/{video_name[:-4]}_{video_id}.csv")

    axRight.set_xlim(0, width)
    axRight.set_ylim(0, height)
    axRight.invert_yaxis()
    axRight.xaxis.tick_top()

    currFrame = 0
    video_path = f"/notebooks/data/Datasets/CPTAD/Videos/{video_name}"
    cap = cv2.VideoCapture(video_path)
    videoSlider.valmax = length-1
    videoSlider.ax.set_xlim(videoSlider.valmin, videoSlider.valmax)

    preCrashBoxes = calcPreCrash(tracks[0]) 

def plotCrashVolumes():
    global tracks, axLeft, fig, length

    # Remove all existing patches before plotting new ones
    [p.remove() for p in reversed(axLeft.patches)]

    grouped_results = []
    iou_list = []
    for framenum in range(length):

        pxtl = None
        for row in preCrashBoxes:
            if row[0] == framenum:
                pxtl = row[1]
                pytl = row[2]
                pxbr = row[3]
                pybr = row[4]
                r = Rectangle((pxtl, pytl), pxbr-pxtl, pybr-pytl, color='purple', alpha=slider1.val)
                axLeft.add_patch(r)
                art3d.pathpatch_2d_to_3d(r, z=framenum, zdir="x")

        crash_frames = tracks.findall(f".//track[@Element1='Car'][@Element2='Car']/frame[@frame='{framenum}']")
        for crash_frame in crash_frames:
            xtl, ytl, xbr, ybr = getAnnoBox(crash_frame)
            r = Rectangle((xtl, ytl), xbr-xtl, ybr-ytl, color='r', alpha=slider1.val)
            axLeft.add_patch(r)
            art3d.pathpatch_2d_to_3d(r, z=framenum, zdir="x")

        yolo_filtered = yolo_boxes2.loc[(yolo_boxes2['frame'] == framenum) & (yolo_boxes2['conf'] > MIN_CONF)]
        for j in range(len(yolo_filtered)):
            row = yolo_filtered.iloc[j]
            if row['conf'] < confSlider2.val:
                continue
            xtl, ytl, xbr, ybr = getResBox(row)
            r = Rectangle((xtl, ytl), xbr-xtl, ybr-ytl, color='c', alpha=slider2.val*row['conf'])
            axLeft.add_patch(r)
            art3d.pathpatch_2d_to_3d(r, z=framenum+0.2, zdir="x")

        yolo_filtered = yolo_boxes4.loc[(yolo_boxes4['frame'] == framenum) & (yolo_boxes4['conf'] > MIN_CONF)]
        for j in range(len(yolo_filtered)):
            row = yolo_filtered.iloc[j]
            if row['conf'] < confSlider3.val:
                continue
            xtl, ytl, xbr, ybr = getResBox(row)
            r = Rectangle((xtl, ytl), xbr-xtl, ybr-ytl, color='y', alpha=slider3.val*row['conf'])
            axLeft.add_patch(r)
            art3d.pathpatch_2d_to_3d(r, z=framenum+0.1, zdir="x")

            box_iou = 0
            if pxtl is not None:
                box_iou = max(iou(getResBox(row), [pxtl, pytl, pxbr, pybr]), box_iou)

            for crash_frame in crash_frames:
                box_iou = max(iou(getResBox(row), getAnnoBox(crash_frame)), box_iou)

            iou_list.append(box_iou)
            print(f"Frame: {framenum}   IOU: {box_iou}")

    avg_iou = 0
    if len(iou_list) > 0:
        avg_iou = np.average(iou_list)
    print(f"Number of Detections: {len(iou_list)}   Average IOU: {avg_iou}")
    fig.canvas.draw()

def plotVideoFrame(val):
    global axRight, cap, currFrame, height

    axRight.cla()

    currFrame = int(val)
    cap.set(cv2.CAP_PROP_POS_FRAMES, currFrame)
    success, frame = cap.read()

    if not success:
        print('Cap Read Error')
        return
    
    for row in preCrashBoxes:
        framenum = row[0]
        if framenum == currFrame:
            xtl = row[1]
            ytl = row[2]
            xbr = row[3]
            ybr = row[4]
            cv2.rectangle(frame, (int(round(xtl)), int(round(ytl))), (int(round(xbr)), int(round(ybr))), (0x7e, 0x1e, 0x9c), 2)

    crash_frames = tracks.findall(f".//track[@Element1='Car'][@Element2='Car']/frame[@frame='{currFrame}']")
    for crash_frame in crash_frames:
        xtl, ytl, xbr, ybr = getAnnoBox(crash_frame)
        cv2.rectangle(frame, (int(round(xtl)), int(round(ytl))), (int(round(xbr)), int(round(ybr))), (255, 0, 0), 2)

    yolo_filtered = yolo_boxes2.loc[(yolo_boxes2['frame'] == currFrame) & (yolo_boxes2['conf'] > MIN_CONF)]
    for j in range(len(yolo_filtered)):
        row = yolo_filtered.iloc[j]
        if row['conf'] < confSlider2.val:
            continue
        xtl, ytl, xbr, ybr = getResBox(row)
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0,255,255), 2)
        cv2.putText(frame, f"{row['conf']:.2f}", (xtl, ybr), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    
    yolo_filtered = yolo_boxes4.loc[(yolo_boxes4['frame'] == currFrame) & (yolo_boxes4['conf'] > MIN_CONF)]
    for j in range(len(yolo_filtered)):
        row = yolo_filtered.iloc[j]
        if row['conf'] < confSlider3.val:
            continue
        xtl, ytl, xbr, ybr = getResBox(row)
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (255,255,0), 2)
        cv2.putText(frame, f"{row['conf']:.2f}", (xtl, ytl), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    # Display the annotated frame
    cv2.putText(frame, f"{currFrame}", (0, height), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    axRight.imshow(frame)

tree = ET.parse('/notebooks/Thesis/1_Preprocessing/annotations_reformatted.xml')
videos = tree.getroot()

video_list = pd.read_csv('/notebooks/Thesis/data_yolo/test_videos.txt', sep=" ", header=None)
idx = 0
video_id = video_list.iloc[idx, 0]
video_name = video_list.iloc[idx, 1]
yolo_boxes2 = None
yolo_boxes4 = None
tracks = None
length = 10
height = 10
currFrame = 0
preCrashBoxes = []

norm_video_list = os.listdir('/notebooks/data/Datasets/CPTAD/Videos_Normal/Test/')
norm_idx = 0

fig = plt.figure("GUI", figsize=(16, 9))

axLeft = fig.add_subplot([0, 0.3, 0.5, 0.7], projection='3d')
axRight = fig.add_subplot([0.5, 0.3, 0.45, 0.65])

# Create function to be called when slider value is changed
def update(val):
    plotCrashVolumes()

# Create 3 axes for 3 sliders red,green and blue
axSlider1 = plt.axes([0.04, 0.20, 0.17, 0.03])
axSlider2 = plt.axes([0.04, 0.17, 0.17, 0.03])
axSlider3 = plt.axes([0.04, 0.14, 0.17, 0.03])

# Create a slider from 0.0 to 1.0 in axes axSlider1
# with 0.6 as initial value.
slider1 = Slider(axSlider1, 'Anno', 0.0, 0.3, valinit=0.15)
slider2 = Slider(axSlider2, 'Yolo2', 0.0, 1.0, valinit=0.0)
slider3 = Slider(axSlider3, 'Yolo4', 0.0, 1.0, valinit=1.0)
 
# Call update function when slider value is changed
slider1.on_changed(update)
slider2.on_changed(update)
slider3.on_changed(update)

axConfSlider2 = plt.axes([0.25, 0.17, 0.17, 0.03])
axConfSlider3 = plt.axes([0.25, 0.14, 0.17, 0.03])

confSlider2 = Slider(axConfSlider2, '', 0, 1.0, valinit=0.8)
confSlider3 = Slider(axConfSlider3, '', 0, 1.0, valinit=0.8)

confSlider2.on_changed(update)
confSlider3.on_changed(update)
 
# Create axes for next video button and create button
axNextVideoButton = plt.axes([0.84, 0.025, 0.1, 0.04])
axPrevVideoButton = plt.axes([0.54, 0.025, 0.1, 0.04])
nextVideoButton = Button(axNextVideoButton, 'Next Crash Video', color='gold',
                hovercolor='skyblue')
prevVideoButton = Button(axPrevVideoButton, 'Prev Crash Video', color='gold',
                hovercolor='skyblue')
 
def nextVideo(event):
    global idx, video_id, video_name, currFrame
    idx = min(idx+1, len(video_list)-1) 
    video_id = video_list.iloc[idx, 0]
    video_name = video_list.iloc[idx, 1]
    currFrame = 0
    loadVideoResults()
    plotCrashVolumes()
    prevFrame(None)

def prevVideo(event):
    global idx, video_id, video_name, currFrame
    idx = max(idx-1, 0)
    video_id = video_list.iloc[idx, 0]
    video_name = video_list.iloc[idx, 1]
    currFrame = 0
    loadVideoResults()
    plotCrashVolumes()
    prevFrame(None)

# Call resetSlider function when clicked on reset button
nextVideoButton.on_clicked(nextVideo)
prevVideoButton.on_clicked(prevVideo)

# Create axes for next video button and create button
axNextNormVideoButton = plt.axes([0.84, 0.07, 0.1, 0.04])
axPrevNormVideoButton = plt.axes([0.54, 0.07, 0.1, 0.04])
nextNormVideoButton = Button(axNextNormVideoButton, 'Next Norm Video', color='gold',
                hovercolor='skyblue')
prevNormVideoButton = Button(axPrevNormVideoButton, 'Prev Norm Video', color='gold',
                hovercolor='skyblue')

def nextNormVideo(event):
    global norm_idx, currFrame
    norm_idx = min(norm_idx+1, len(norm_video_list)-1) 
    currFrame = 0
    loadNormalResults(norm_idx)
    plotCrashVolumes()
    prevFrame(None)

def prevNormVideo(event):
    global norm_idx, currFrame
    norm_idx = max(norm_idx-1, 0)
    currFrame = 0
    loadNormalResults(norm_idx)
    plotCrashVolumes()
    prevFrame(None)

# Call resetSlider function when clicked on reset button
nextNormVideoButton.on_clicked(nextNormVideo)
prevNormVideoButton.on_clicked(prevNormVideo)

axVideoSlider = plt.axes([0.54, 0.20, 0.4, 0.03])
videoSlider = Slider(axVideoSlider, 'Frame', 0, length, valinit=0, valstep=1)
videoSlider.on_changed(plotVideoFrame)

axNextFrameButton = plt.axes([0.84, 0.15, 0.1, 0.04])
axPrevFrameButton = plt.axes([0.54, 0.15, 0.1, 0.04])
nextFrameButton = Button(axNextFrameButton, 'Next Frame', color='gold',
                hovercolor='skyblue')
prevFrameButton = Button(axPrevFrameButton, 'Prev Frame', color='gold',
                hovercolor='skyblue')

def nextFrame(event):
    global currFrame, axVideoSlider
    currFrame = min(currFrame+1, length-1)
    videoSlider.set_val(currFrame)
    plotVideoFrame(currFrame)

def prevFrame(event):
    global currFrame, axVideoSlider
    currFrame = max(currFrame-1, 0)
    videoSlider.set_val(currFrame)
    plotVideoFrame(currFrame)

nextFrameButton.on_clicked(nextFrame)
prevFrameButton.on_clicked(prevFrame)

loadVideoResults()
plotCrashVolumes()
plotVideoFrame(0)
plt.show()