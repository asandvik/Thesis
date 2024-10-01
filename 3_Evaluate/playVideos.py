import os
import cv2
import pandas as pd
from time import sleep
from ultralytics import YOLO
import xml.etree.ElementTree as ET

NORMAL = False

INCL_ANNO = True
INCL_YOLO2 = True
INCL_YOLO4 = True
INCL_R3p1 = False

# no point in including annotations on normal video
INCL_ANNO = INCL_ANNO and not NORMAL

MIN_CONF = 0.25

# Open video list
if NORMAL:
    video_list = os.listdir('/notebooks/data/Datasets/CPTAD/Videos_Normal/Test/')
else:
    video_list = pd.read_csv('/notebooks/Thesis/data_yolo/test_videos.txt', sep=" ", header=None)

# Open annotations
if INCL_ANNO:
    tree = ET.parse('/notebooks/Thesis/1_Preprocessing/annotations_reformatted.xml')
    videos = tree.getroot()

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', 1280, 720)

for i in range(len(video_list)):

    # get video path
    if NORMAL:
        video_name = video_list[i]
        video_path = f"/notebooks/data/Datasets/CPTAD/Videos_Normal/Test/{video_name}"
    else:
        video_id = video_list.iloc[i, 0]
        video_name = video_list.iloc[i, 1]
        # video_id = 437028
        # video_name = 'tDN-mwNSJc804.mp4'
        video_path = f"/notebooks/data/Datasets/CPTAD/Videos/{video_name}"

    # get tracks from annotations
    if INCL_ANNO:
        video = videos.find(f".//video[@taskid='{video_id}']")
        tracks = video.find('tracks')

    if INCL_YOLO4:
        yolo_boxes4 = pd.read_csv(f"/notebooks/Thesis/results/yolo_train4/{video_name[:-4]}_{video_id}.csv")

    if INCL_YOLO2:
        yolo_boxes2 = pd.read_csv(f"/notebooks/Thesis/results/yolo_train2/{video_name[:-4]}_{video_id}.csv")

    print(video_path)    
    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        framenum = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        success, frame = cap.read()

        if success:
 
            # show annotations
            if INCL_ANNO:
                crash_frames = tracks.findall(f".//track[@Element1='Car'][@Element2='Car']/frame[@frame='{framenum}']")
                for crash_frame in crash_frames:
                    xtl = int(round(float(crash_frame.attrib.get('xtl'))))
                    ytl = int(round(float(crash_frame.attrib.get('ytl'))))
                    xbr = int(round(float(crash_frame.attrib.get('xbr'))))
                    ybr = int(round(float(crash_frame.attrib.get('ybr'))))
                    cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0,0,255), 2)

            if INCL_YOLO4:
                yolo_filtered = yolo_boxes4.loc[(yolo_boxes4['frame'] == framenum) & (yolo_boxes4['conf'] > MIN_CONF)]
                for j in range(len(yolo_filtered)):
                    row = yolo_filtered.iloc[j]
                    xtl = round(row['x'] - row['w']/2)
                    ytl = round(row['y'] - row['h']/2)
                    xbr = round(row['x'] + row['w']/2)
                    ybr = round(row['y'] + row['h']/2)
                    cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0,255,255), 2)
                    cv2.putText(frame, f"{row['conf']:.2f}", (xtl, ytl), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            if INCL_YOLO2:
                yolo_filtered = yolo_boxes2.loc[(yolo_boxes2['frame'] == framenum) & (yolo_boxes2['conf'] > MIN_CONF)]
                for j in range(len(yolo_filtered)):
                    row = yolo_filtered.iloc[j]
                    xtl = round(row['x'] - row['w']/2)
                    ytl = round(row['y'] - row['h']/2)
                    xbr = round(row['x'] + row['w']/2)
                    ybr = round(row['y'] + row['h']/2)
                    cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (255,255,0), 2)
                    cv2.putText(frame, f"{row['conf']:.2f}", (xtl, ybr), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            # Display the annotated frame
            cv2.putText(frame, f"{framenum}", (0, height), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow("Video", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            sleep(0.03)
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()

cv2.destroyAllWindows()