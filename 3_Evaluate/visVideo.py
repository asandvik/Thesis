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

def IOminarea(boxA, boxB):
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

def IOU(boxA, boxB):
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
    global width, height

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

    for i in range(1,30):

        xtl -= dxtl
        ytl -= dytl
        xbr -= dxbr
        ybr -= dybr

        # stay within bounds of frame
        xtl = max(0, xtl)
        xtl = min(xtl, width)
        ytl = max(0, ytl)
        ytl = min(ytl, height)
        xbr = max(0, xbr)
        xbr = min(xbr, width)
        ybr = max(0, ybr)
        ybr = min(ybr, height)

        # stop if extrapolated boxes cross in on themselves
        if xtl >= xbr or ytl >= ybr:
            break

        data.append([frame-i, xtl, ytl, xbr, ybr])

    return np.array(data)

def loadNormalResults(i):
    global norm_video_list, cap, yolo_boxes2, yolo_boxes4, tracks, preCrashBoxes, axLeft, height, width, length, video_name

    video_name = norm_video_list[i]

    plt.suptitle(video_name)

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
    global axRight, length, cap, videoSlider, currFrame, height, length, width
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
    global tracks, axLeft, fig, length, yolo_colors

    # to update yolo_colors
    filterYolo(length, preCrashBoxes, yolo_boxes4, tracks)

    # Remove all existing patches before plotting new ones
    [p.remove() for p in reversed(axLeft.patches)]

    color_idx = 0
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

        # yolo_filtered = yolo_boxes2.loc[(yolo_boxes2['frame'] == framenum) & (yolo_boxes2['conf'] > confSlider2.val)]
        # for j in range(len(yolo_filtered)):
        #     row = yolo_filtered.iloc[j]
        #     xtl, ytl, xbr, ybr = getResBox(row)
        #     r = Rectangle((xtl, ytl), xbr-xtl, ybr-ytl, color='c', alpha=slider2.val*row['conf'])
        #     axLeft.add_patch(r)
        #     art3d.pathpatch_2d_to_3d(r, z=framenum+0.2, zdir="x")

        yolo_filtered = yolo_boxes4.loc[(yolo_boxes4['frame'] == framenum) & (yolo_boxes4['conf'] > confSlider3.val)]
        for j in range(len(yolo_filtered)):
            row = yolo_filtered.iloc[j]
            xtl, ytl, xbr, ybr = getResBox(row)

            box_iou = 0
            if pxtl is not None:
                box_iou = max(IOU(getResBox(row), [pxtl, pytl, pxbr, pybr]), box_iou)

            for crash_frame in crash_frames:
                box_iou = max(IOU(getResBox(row), getAnnoBox(crash_frame)), box_iou)

            color = yolo_colors[color_idx][0]
            color_idx += 1
            # if box_iou >= annoIOUslider.val:
            #     color = 'y'
            # else:
            #     color = 'gray'

            r = Rectangle((xtl, ytl), xbr-xtl, ybr-ytl, color=color, alpha=slider3.val*row['conf'])
            axLeft.add_patch(r)
            art3d.pathpatch_2d_to_3d(r, z=framenum+0.1, zdir="x")

    fig.canvas.draw()

def filterYolo(length, preCrashBoxes, yolo_boxes4, tracks):
    global yolo_colors
    
    # TODO: color list for TPFN for yolo results. needs to pass into crash volumes and cv2 frame

    ideal_signal = []

    naive_signal = []
    naive_sig = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    detect_delay = None

    alarm_signal = []
    alarm_sig = 0
    ftp = 0
    ffp = 0
    ftn = 0
    ffn = 0
    fdetect_delay = None

    if len(tracks) > 0: # if crash video
        impact_frame = int(tracks[0].attrib.get('start'))
    
    yolo_colors = []
    yolo_alarm_scores = []
    for framenum in range(length):

        # PreCrash Slack
        pxtl = None
        for row in preCrashBoxes:
            if row[0] == framenum:
                pxtl = row[1]
                pytl = row[2]
                pxbr = row[3]
                pybr = row[4]

        # Ground truth annotations
        crash_frames = tracks.findall(f".//track[@Element1='Car'][@Element2='Car']/frame[@frame='{framenum}']")
        
        # Model Predictions
        yolo_filtered = yolo_boxes4.loc[(yolo_boxes4['frame'] == framenum) & (yolo_boxes4['conf'] > confSlider3.val)]
        prev_yolo_filtered = yolo_boxes4.loc[(yolo_boxes4['frame'] == framenum-1) & (yolo_boxes4['conf'] > confSlider3.val)]
        prev_yolo_alarm_scores = yolo_alarm_scores
        yolo_alarm_scores = []

        naive_sig = 1 if len(yolo_filtered) > 0 else 0
        ideal_sig = 0

        expected = 1 if len(crash_frames) > 0 and len(yolo_filtered) == 0 else 0

        # if len(prev_yolo_filtered) == 0:
        #     alarm_sig = 0 # enforce consecutive

        predicted = 0
        fpredicted = 0
        for j in range(len(yolo_filtered)):
            predicted = 1
            row = yolo_filtered.iloc[j]

            # get overlap with with annotations
            ideal_sig = 0
            box_iou = 0
            if pxtl is not None:
                box_iou = max(IOU(getResBox(row), [pxtl, pytl, pxbr, pybr]), box_iou)

            for crash_frame in crash_frames:
                box_iou = max(IOU(getResBox(row), getAnnoBox(crash_frame)), box_iou)

            # preprocessed assessment -> just minConf
            if box_iou >= annoIOUslider.val:
                expected = 1 # set to 1 if the overlap between detection and annotation is high enough
                ideal_sig = 1
                if detect_delay is None:
                    detect_delay = framenum - impact_frame

            # postprocessed -> consecutive enforcement
            consecutive_iou = 0
            idx_prev_detection = None
            for k in range(len(prev_yolo_filtered)):
                prev_row = prev_yolo_filtered.iloc[k]
                curr_iou = IOU(getResBox(row), getResBox(prev_row))
                if curr_iou > consecutive_iou:
                    consecutive_iou = curr_iou
                    idx_prev_detection = k

            if idx_prev_detection is None:
                box_alarm_score = 0
            elif consIOUMinslider.val <= consecutive_iou and consecutive_iou <= consIOUMaxslider.val:
                box_alarm_score = 1 + prev_yolo_alarm_scores[idx_prev_detection]
            else:        
                box_alarm_score = 0    
            yolo_alarm_scores.append(box_alarm_score)

            # postproccessed assessment -> include consecutive enforcement
            if box_iou >= annoIOUslider.val:
                if box_alarm_score >= alarmThreshSlider.val: # raised filtered true positive
                    yolo_colors.append(['y', (255, 255, 0)])
                    if fdetect_delay is None:
                        fdetect_delay = framenum - impact_frame
                else: # ignored filtered true positive
                    yolo_colors.append(['gray', (0, 0, 0)])
            else: 
                if box_alarm_score >= alarmThreshSlider.val: # raised filtered false positive
                    yolo_colors.append(['b', (0, 0, 255)])
                else: # ignored filtered false positive
                    yolo_colors.append(['gray', (0, 0, 0)])

        if len(yolo_alarm_scores) > 0:
            alarm_sig = max(yolo_alarm_scores)
        else:
            alarm_sig = 0
        
        if len(yolo_filtered) > 0 and alarm_sig >= alarmThreshSlider.val:
            fpredicted = 1
        
        # print("new scores", yolo_alarm_scores)

        if expected == 0 and predicted == 0:
            tn += 1
        elif expected == 1 and predicted == 0:
            fn += 1
        elif expected == 0 and predicted == 1:
            fp += 1
        elif expected == 1 and predicted == 1:
            tp += 1
        else:
            raise ValueError("expected and predicted must be 0 or 1")
        
        if expected == 0 and fpredicted == 0:
            ftn += 1
        elif expected == 1 and fpredicted == 0:
            ffn += 1
        elif expected == 0 and fpredicted == 1:
            ffp += 1
        elif expected == 1 and fpredicted == 1:
            ftp += 1
        else:
            raise ValueError("expected and fpredicted must be 0 or 1")

        ideal_signal.append(ideal_sig)
        alarm_signal.append(alarm_sig)
        naive_signal.append(naive_sig)

    return [[tp, fp, tn, fn, detect_delay], 
            [ftp, ffp, ftn, ffn, fdetect_delay], 
            ideal_signal, naive_signal, alarm_signal]

# Widget Callbacks

def cbPlotVideoFrame(val):
    global axRight, cap, currFrame, height, yolo_colors

    axRight.cla()

    currFrame = int(val)
    cap.set(cv2.CAP_PROP_POS_FRAMES, currFrame)
    success, frame = cap.read()

    if not success:
        print('Cap Read Error')
        return
   
    pxtl = None
    for row in preCrashBoxes:
        framenum = row[0]
        if framenum == currFrame:
            pxtl = row[1]
            pytl = row[2]
            pxbr = row[3]
            pybr = row[4]
            cv2.rectangle(frame, (int(round(pxtl)), int(round(pytl))), (int(round(pxbr)), int(round(pybr))), (0x7e, 0x1e, 0x9c), 2)

    crash_frames = tracks.findall(f".//track[@Element1='Car'][@Element2='Car']/frame[@frame='{currFrame}']")
    for crash_frame in crash_frames:
        xtl, ytl, xbr, ybr = getAnnoBox(crash_frame)
        cv2.rectangle(frame, (int(round(xtl)), int(round(ytl))), (int(round(xbr)), int(round(ybr))), (255, 0, 0), 2)

    # yolo_filtered = yolo_boxes2.loc[(yolo_boxes2['frame'] == currFrame) & (yolo_boxes2['conf'] > confSlider2.val)]
    # for j in range(len(yolo_filtered)):
    #     row = yolo_filtered.iloc[j]
    #     xtl, ytl, xbr, ybr = getResBox(row)
    #     cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0,255,255), 2)
    #     cv2.putText(frame, f"{row['conf']:.2f}", (xtl, ybr), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    
    #color_idx = 0
    yolo_filtered = yolo_boxes4.loc[(yolo_boxes4['frame'] == currFrame) & (yolo_boxes4['conf'] > confSlider3.val)]
    for j in range(len(yolo_filtered)):
        row = yolo_filtered.iloc[j]
        xtl, ytl, xbr, ybr = getResBox(row)

        box_iou = 0
        if pxtl is not None:
            box_iou = max(IOU(getResBox(row), [pxtl, pytl, pxbr, pybr]), box_iou)

        for crash_frame in crash_frames:
            box_iou = max(IOU(getResBox(row), getAnnoBox(crash_frame)), box_iou)

        # doesn't work because not guaranteed to access sequentially. Would need an id for each detection
        # color = yolo_colors[color_idx][1]
        # color_idx += 1

        if box_iou >= annoIOUslider.val:
            color = (255, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), color, 2)
        cv2.putText(frame, f"{row['conf']:.2f}", (xtl, ytl), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display the annotated frame
    cv2.putText(frame, f"{currFrame}", (0, height), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    axRight.imshow(frame)

def cbUpdate(val):
    plotCrashVolumes()

def cbNextVideo(event):
    global idx, video_id, video_name, currFrame
    idx = min(idx+1, len(video_list)-1) 
    video_id = video_list.iloc[idx, 0]
    video_name = video_list.iloc[idx, 1]
    currFrame = 0
    textbox.text_disp.set_text(f"{idx}")
    loadVideoResults()
    plotCrashVolumes()
    cbPrevFrame(None)
    cbFilterCurrResult(None)

def cbPrevVideo(event):
    global idx, video_id, video_name, currFrame
    idx = max(idx-1, 0)
    video_id = video_list.iloc[idx, 0]
    video_name = video_list.iloc[idx, 1]
    currFrame = 0
    textbox.text_disp.set_text(f"{idx}")
    loadVideoResults()
    plotCrashVolumes()
    cbPrevFrame(None)
    cbFilterCurrResult(None)

def cbNextNormVideo(event):
    global norm_idx, currFrame
    norm_idx = min(norm_idx+1, len(norm_video_list)-1) 
    currFrame = 0
    textboxNorm.text_disp.set_text(f"{norm_idx}")
    loadNormalResults(norm_idx)
    plotCrashVolumes()
    cbPrevFrame(None)
    cbFilterCurrResult(None)

def cbPrevNormVideo(event):
    global norm_idx, currFrame
    norm_idx = max(norm_idx-1, 0)
    currFrame = 0
    textboxNorm.text_disp.set_text(f"{norm_idx}")
    loadNormalResults(norm_idx)
    plotCrashVolumes()
    cbPrevFrame(None)
    cbFilterCurrResult(None)

def cbNextFrame(event):
    global currFrame, axVideoSlider
    currFrame = min(currFrame+1, length-1)
    videoSlider.set_val(currFrame)
    cbPlotVideoFrame(currFrame)

def cbPrevFrame(event):
    global currFrame, axVideoSlider
    currFrame = max(currFrame-1, 0)
    videoSlider.set_val(currFrame)
    cbPlotVideoFrame(currFrame)

def submit(text):
    global idx, video_id, video_name, currFrame

    try:
        i = int(text)
    except:
        return
    if i >= len(video_list):
        return
    idx = i
    video_id = video_list.iloc[idx, 0]
    video_name = video_list.iloc[idx, 1]
    currFrame = 0
    loadVideoResults()
    plotCrashVolumes()
    cbPrevFrame(None)

def submitNorm(text):
    global norm_idx, currFrame

    try:
        i = int(text)
    except:
        return
    if i >= len(norm_video_list):
        return
    norm_idx = i
    currFrame = 0
    loadNormalResults(norm_idx)
    plotCrashVolumes()
    cbPrevFrame(None)

# def plotAlarm(signal1, signal2, signal3):
#     global axMiddleTop, axMiddleMid, axMiddleBot, axMiddleBot2
#     axMiddleTop.cla()
#     axMiddleMid.cla()
#     axMiddleBot.cla()
#     axMiddleBot2.cla()
#     axMiddleTop.set_ylabel("Naive")
#     axMiddleMid.set_ylabel("Ideal")
#     axMiddleBot.set_ylabel("Filtered")
#     axMiddleTop.plot(signal2)
#     axMiddleMid.plot(signal1)

#     binary_signal = []
#     for sample in signal3:
#         sample = 1 if sample > alarmThreshSlider.val else 0
#         binary_signal.append(sample)
#     axMiddleBot2.plot(binary_signal, color='r')
#     axMiddleBot.plot(signal3)

#     axMiddleTop.set_ylim(bottom=0)
#     axMiddleMid.set_ylim(bottom=0)
#     axMiddleBot.set_ylim(bottom=0)
#     axMiddleBot2.set_ylim(bottom=0)

#     fig.canvas.draw()

def printMetrics(res):
    tp, fp, tn, fn, dd = res

    # accuracy
    a = (tp + tn) / (tp + fp + tn + fn)

    # precision
    if (tp + fp) > 0:
        p = tp / (tp + fp)
    else:
        p = 0

    # recall
    if (tp + fn) > 0:
        r = tp / (tp + fn) 
    else:
        r = 0

    s = tn / (tn + fp) # specificity

    print(f"  A={a:.4f} P={p:0.4f} R={r:0.4f} S={s:0.4f} TP={tp:>3} FP={fp:>3} TN={tn:>3} FN={fn:>3} DD={str(dd):>4}")

    return [a, p, r, s] 
    
def printRes(idx, name, result):
    pre = result[0]
    post = result[1]
    print(f"{idx:2} {name}")
    preAPRS = printMetrics(pre)
    postAPRS = printMetrics(post)
    return [preAPRS, postAPRS]
    

def cbFilterCurrResult(event):
    global video_name, length, preCrashBoxes, yolo_boxes4, tracks
    result = filterYolo(length, preCrashBoxes, yolo_boxes4, tracks)
    printRes(99, video_name, result)
    # plotAlarm(result[2], result[3], result[4])

def cbFilterAllResults(event):
    global video_list, norm_video_list

    pre_scores = []
    post_scores = []
    pre_dds = [] # detection delay
    post_dds = []
    pre_results = []
    post_results = []
    for i in range(len(video_list)):
        video_id = video_list.iloc[i, 0]
        video_name = video_list.iloc[i, 1]
        video = videos.find(f".//video[@taskid='{video_id}']")
        length = int(video.find('length').text)
        tracks = video.find('tracks')
        preCrashBoxes = calcPreCrash(tracks[0])
        yolo_boxes4 = pd.read_csv(f"/notebooks/Thesis/results/yolo_train4/{video_name[:-4]}_{video_id}.csv")
        result = filterYolo(length, preCrashBoxes, yolo_boxes4, tracks)
        aprs = printRes(i, video_name, result)
        pre_score = aprs[0][1] # use precision
        post_score = aprs[1][1] # use precision
        pre_scores.append(pre_score)
        post_scores.append(post_score)
        pre_results.append(aprs[0])
        post_results.append(aprs[1])
        pre_dd = result[0][4]
        post_dd = result[1][4]
        pre_dds.append(pre_dd) if pre_dd is not None else None
        post_dds.append(post_dd) if post_dd is not None else None

    pre_norm_scores = []
    post_norm_scores = []
    pre_norm_results = []
    post_norm_results = []
    for i in range(len(norm_video_list)):
        video_name = norm_video_list[i]
        video_path = f"/notebooks/data/Datasets/CPTAD/Videos_Normal/Test/{video_name}"
        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        yolo_boxes4 = pd.read_csv(f"/notebooks/Thesis/results/yolo_train4_norm/{video_name[:-4]}_{i}.csv")
        preCrashBoxes = []
        tracks = ET.Element("Empty")
        result = filterYolo(length, preCrashBoxes, yolo_boxes4, tracks)
        aprs = printRes(i, video_name, result)
        pre_score = aprs[0][3] # use specificity
        post_score = aprs[1][3] # use specificity
        pre_norm_scores.append(pre_score)
        post_norm_scores.append(post_score)
        pre_norm_results.append(aprs[0])
        post_norm_results.append(aprs[1])

    print("All")

    print("Crashes:")
    npResults = np.array(pre_results)
    a_avg = np.average(npResults[:, 0])
    p_avg = np.average(npResults[:, 1])
    r_avg = np.average(npResults[:, 2])
    s_avg = np.average(npResults[:, 3])
    print(f" Preprocessed:  A={a_avg:.4f} P={p_avg:.4f} R={r_avg:.4f} S={s_avg:.4f}")
    npResults = np.array(post_results)
    a_avg = np.average(npResults[:, 0])
    p_avg = np.average(npResults[:, 1])
    r_avg = np.average(npResults[:, 2])
    s_avg = np.average(npResults[:, 3])
    print(f" Postprocessed: A={a_avg:.4f} P={p_avg:.4f} R={r_avg:.4f} S={s_avg:.4f}")

    print("Normal:")
    npResults = np.array(pre_norm_results)
    a_avg = np.average(npResults[:, 0])
    p_avg = np.average(npResults[:, 1])
    r_avg = np.average(npResults[:, 2])
    s_avg = np.average(npResults[:, 3])
    print(f" Preprocessed:  A={a_avg:.4f} P={p_avg:.4f} R={r_avg:.4f} S={s_avg:.4f}")
    npResults = np.array(post_norm_results)
    a_avg = np.average(npResults[:, 0])
    p_avg = np.average(npResults[:, 1])
    r_avg = np.average(npResults[:, 2])
    s_avg = np.average(npResults[:, 3])
    print(f" Postprocessed: A={a_avg:.4f} P={p_avg:.4f} R={r_avg:.4f} S={s_avg:.4f}")

    ax2Top.cla()
    ax2Mid.cla()
    ax2Bot.cla()

    nbins = 50
    bins = np.histogram(np.hstack((pre_scores, post_scores)), bins=nbins)[1]
    ax2Top.hist(pre_scores, bins=bins, alpha=0.5, label=f"Pre  ({len(pre_scores)})", color='blue')
    ax2Top.hist(post_scores, bins=bins, alpha=0.5, label=f"Post ({len(post_scores)})", color='orange')
    ax2Top.legend()
    ax2Top.set_title('Crash Video Scores (Precision)')

    bins = np.histogram(np.hstack((pre_dds, post_dds)), bins=nbins)[1]
    ax2Mid.hist(pre_dds, bins=bins, alpha=0.5, label=f"Pre  ({len(pre_dds)})", color='blue')
    ax2Mid.hist(post_dds, bins=bins, alpha=0.5, label=f"Post ({len(post_dds)})", color='orange')
    ax2Mid.legend()
    ax2Mid.set_title('Crash Videos Detection Delay (Frames)')

    bins = np.histogram(np.hstack((pre_norm_scores, post_norm_scores)), bins=nbins)[1]
    ax2Bot.hist(pre_norm_scores, bins=bins, alpha=0.5, label=f"Pre  ({len(pre_norm_scores)})", color='blue')
    ax2Bot.hist(post_norm_scores, bins=bins, alpha=0.5, label=f"Post ({len(post_norm_scores)})", color='orange')
    ax2Bot.legend()
    ax2Bot.set_title('Normal Video Scores (Specificity)')

    fig2.canvas.draw()


def cbFilterAllResults2(event):
    global video_list, norm_video_list

    threshlist = [x / 100 for x in range(0, 101, 1)]

    print("Crashes:")

    for thresh in threshlist:
        
        consIOUMinslider.set_val(thresh)

        pre_scores = []
        post_scores = []
        pre_dds = [] # detection delay
        post_dds = []
        pre_results = []
        post_results = []
        for i in range(len(video_list)):
            video_id = video_list.iloc[i, 0]
            video_name = video_list.iloc[i, 1]
            video = videos.find(f".//video[@taskid='{video_id}']")
            length = int(video.find('length').text)
            tracks = video.find('tracks')
            preCrashBoxes = calcPreCrash(tracks[0])
            yolo_boxes4 = pd.read_csv(f"/notebooks/Thesis/results/yolo_train4/{video_name[:-4]}_{video_id}.csv")
            result = filterYolo(length, preCrashBoxes, yolo_boxes4, tracks)
            aprs = printRes(i, video_name, result)
            pre_score = aprs[0][1] # use precision
            post_score = aprs[1][1] # use precision
            pre_scores.append(pre_score)
            post_scores.append(post_score)
            pre_results.append(aprs[0])
            post_results.append(aprs[1])
            pre_dd = result[0][4]
            post_dd = result[1][4]
            pre_dds.append(pre_dd) if pre_dd is not None else None
            post_dds.append(post_dd) if post_dd is not None else None

        npResults = np.array(post_results)
        a_avg = np.average(npResults[:, 0])
        p_avg = np.average(npResults[:, 1])
        r_avg = np.average(npResults[:, 2])
        s_avg = np.average(npResults[:, 3])
        print(f"{thresh},{a_avg:.4f},{p_avg:.4f},{r_avg:.4f},{s_avg:.4f}")

    print("Normal")

    for thresh in threshlist:

        consIOUMinslider.set_val(thresh)

        pre_norm_scores = []
        post_norm_scores = []
        pre_norm_results = []
        post_norm_results = []
        for i in range(len(norm_video_list)):
            video_name = norm_video_list[i]
            video_path = f"/notebooks/data/Datasets/CPTAD/Videos_Normal/Test/{video_name}"
            cap = cv2.VideoCapture(video_path)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            yolo_boxes4 = pd.read_csv(f"/notebooks/Thesis/results/yolo_train4_norm/{video_name[:-4]}_{i}.csv")
            preCrashBoxes = []
            tracks = ET.Element("Empty")
            result = filterYolo(length, preCrashBoxes, yolo_boxes4, tracks)
            aprs = printRes(i, video_name, result)
            pre_score = aprs[0][3] # use specificity
            post_score = aprs[1][3] # use specificity
            pre_norm_scores.append(pre_score)
            post_norm_scores.append(post_score)
            pre_norm_results.append(aprs[0])
            post_norm_results.append(aprs[1])

        npResults = np.array(post_norm_results)
        a_avg = np.average(npResults[:, 0])
        p_avg = np.average(npResults[:, 1])
        r_avg = np.average(npResults[:, 2])
        s_avg = np.average(npResults[:, 3])
        print(f"{thresh},{a_avg:.4f},{p_avg:.4f},{r_avg:.4f},{s_avg:.4f}")

def cbSavePlots(event):
    global fig, axLeft, axRight, video_name

    prefix = video_name[:-4]

    extent = axLeft.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"/notebooks/Thesis/3_Evaluate/figures/{prefix}_3dplot.png", bbox_inches=extent.expanded(1.2, 1.1))

    extent = axRight.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"/notebooks/Thesis/3_Evaluate/figures/{prefix}_frame.png", bbox_inches=extent)

    print(f"Saved {prefix}")
    

# Annotations
tree = ET.parse('/notebooks/Thesis/1_Preprocessing/annotations_reformatted.xml')
videos = tree.getroot()

# Video Filename Lists
video_list = pd.read_csv('/notebooks/Thesis/data_yolo/test_videos.txt', sep=" ", header=None)
norm_video_list = os.listdir('/notebooks/data/Datasets/CPTAD/Videos_Normal/Test/')

# Global Variables
preCrashBoxes = []
idx = 0
norm_idx = -1 # -1 bc crash video is displayed first. Prevents skipping first normal video 
video_id = video_list.iloc[idx, 0]
video_name = video_list.iloc[idx, 1]
yolo_boxes2 = None
yolo_boxes4 = None
tracks = None
length = 10
height = 10
width = 10
currFrame = 0
yolo_colors = []

fig2 = plt.figure("Results", figsize=(12, 6))
ax2Top = fig2.add_subplot(2,2,1)
ax2Mid = fig2.add_subplot(2,2,2)
ax2Bot = fig2.add_subplot(2,2,3)
# fig2.subplots_adjust(hspace=0.5)

# GUI Skeleton
fig = plt.figure("GUI", figsize=(16, 6))
# axLeft = fig.add_subplot([0, 0.3, 0.3, 0.7], projection='3d')
# axMiddleTop = fig.add_subplot([0.33, 0.7, 0.3, 0.2])
# axMiddleMid = fig.add_subplot([0.33, 0.5, 0.3, 0.2])
# axMiddleBot = fig.add_subplot([0.33, 0.3, 0.3, 0.2])
# axMiddleBot2 = axMiddleBot.twinx()
# axRight = fig.add_subplot([0.67, 0.3, 0.33, 0.65])

axLeft = fig.add_subplot([0, 0.3, 0.45, 0.7], projection='3d')
axRight = fig.add_subplot([0.55, 0.3, 0.45, 0.65])

# Widget Axes
axSlider1 = plt.axes([0.04, 0.20, 0.17, 0.03])
axSlider2 = plt.axes([0.04, 0.17, 0.17, 0.03])
axSlider3 = plt.axes([0.04, 0.14, 0.17, 0.03])
# axConfSlider2 = plt.axes([0.25, 0.17, 0.17, 0.03])
axConfSlider3 = plt.axes([0.3, 0.20, 0.17, 0.03])
axNextVideoButton = plt.axes([0.84, 0.07, 0.1, 0.04])
axPrevVideoButton = plt.axes([0.54, 0.07, 0.1, 0.04])
axNextNormVideoButton = plt.axes([0.84, 0.025, 0.1, 0.04])
axPrevNormVideoButton = plt.axes([0.54, 0.025, 0.1, 0.04])
axVideoSlider = plt.axes([0.54, 0.20, 0.4, 0.03])
axNextFrameButton = plt.axes([0.84, 0.15, 0.1, 0.04])
axPrevFrameButton = plt.axes([0.54, 0.15, 0.1, 0.04])
axFilterButton = plt.axes([0.4, 0.07, 0.1, 0.04])
axFilterAllButton = plt.axes([0.4, 0.025, 0.1, 0.04])
axTextBox = plt.axes([0.69, 0.07, 0.1, 0.04])
axTextBoxNorm = plt.axes([0.69, 0.025, 0.1, 0.04])
axAnnoIOUSlider = plt.axes([0.04, 0.08, 0.12, 0.03])
axConsIOUMinSlider = plt.axes([0.04, 0.05, 0.12, 0.03])
axConsIOUMaxSlider = plt.axes([0.04, 0.02, 0.12, 0.03])
axAlarmThreshSlider = plt.axes([0.25, 0.08, 0.12, 0.03])

# Widgets
slider1 = Slider(axSlider1, 'Anno', 0.0, 0.04, valinit=0.15)
# slider2 = Slider(axSlider2, 'Yolo2', 0.0, 1.0, valinit=0.0)
slider3 = Slider(axSlider2, 'Yolo4', 0.0, 1.0, valinit=0.5)
# confSlider2 = Slider(axConfSlider2, '', 0, 1.0, valinit=0.8, valstep=0.01)
confSlider3 = Slider(axSlider3, 'MinConf', 0, 1.0, valinit=0.53, valstep=0.01)
nextVideoButton = Button(axNextVideoButton, 'Next Crash Video', color='gold', hovercolor='skyblue')
prevVideoButton = Button(axPrevVideoButton, 'Prev Crash Video', color='gold', hovercolor='skyblue')
nextNormVideoButton = Button(axNextNormVideoButton, 'Next Norm Video', color='gold', hovercolor='skyblue')
prevNormVideoButton = Button(axPrevNormVideoButton, 'Prev Norm Video', color='gold', hovercolor='skyblue')
videoSlider = Slider(axVideoSlider, 'Frame', 0, length, valinit=0, valstep=1)
nextFrameButton = Button(axNextFrameButton, 'Next Frame', color='gold', hovercolor='skyblue')
prevFrameButton = Button(axPrevFrameButton, 'Prev Frame', color='gold', hovercolor='skyblue')
filterButton = Button(axFilterButton, 'Filter Single', color='gold', hovercolor='skyblue')
filterAllButton = Button(axFilterAllButton, 'Filter All', color='gold', hovercolor='skyblue')
textbox = TextBox(axTextBox, 'Index', initial='0')
textboxNorm = TextBox(axTextBoxNorm, 'Index', initial='0')
annoIOUslider = Slider(axConfSlider3, 'AnnoIOU', 0.5, 1.0, valinit=0.56) # using 0.56 bc of crash vid #8 and #10
consIOUMinslider = Slider(axConsIOUMinSlider, 'ConsIOUMin', 0.0, 1.0, valinit=0.86)
consIOUMaxslider = Slider(axConsIOUMaxSlider, 'ConsIOUMax', 0.0, 1.0, valinit=1.0)
alarmThreshSlider = Slider(axAnnoIOUSlider, 'MinConsec', 0.0, 10.0, valinit=1.0, valstep=1)
savePlotsButton = Button(axAlarmThreshSlider, 'Save Figs', color='gold', hovercolor='skyblue')

# Attach Callbacks
slider1.on_changed(cbUpdate)
# slider2.on_changed(cbUpdate)
slider3.on_changed(cbUpdate)
# confSlider2.on_changed(cbUpdate)
confSlider3.on_changed(cbUpdate)
nextVideoButton.on_clicked(cbNextVideo)
prevVideoButton.on_clicked(cbPrevVideo)
nextNormVideoButton.on_clicked(cbNextNormVideo)
prevNormVideoButton.on_clicked(cbPrevNormVideo)
videoSlider.on_changed(cbPlotVideoFrame)
nextFrameButton.on_clicked(cbNextFrame)
prevFrameButton.on_clicked(cbPrevFrame)
filterButton.on_clicked(cbFilterCurrResult)
filterAllButton.on_clicked(cbFilterAllResults)
textbox.on_submit(submit)
textboxNorm.on_submit(submitNorm)
annoIOUslider.on_changed(cbUpdate)
consIOUMinslider.on_changed(cbUpdate)
consIOUMaxslider.on_changed(cbUpdate)
alarmThreshSlider.on_changed(cbUpdate)
savePlotsButton.on_clicked(cbSavePlots)


loadVideoResults()
plotCrashVolumes()
cbPlotVideoFrame(0)
cbFilterCurrResult(None)
plt.show()