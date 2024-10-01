import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import xml.etree.ElementTree as ET

NORMAL = True
ADD_ANNOTATIONS = False
MIN_CONF = 0.25
MODEL = 'train2'

try:
    if NORMAL:
        os.mkdir(f'/notebooks/Thesis/results/yolo_{MODEL}_norm')
    else:
        os.mkdir(f'/notebooks/Thesis/results/yolo_{MODEL}')
except FileExistsError:
    pass

# Load the YOLOv8 model
model = YOLO(f"/notebooks/Thesis/2_Training/runs/detect/{MODEL}/weights/best.pt")

if not NORMAL:
    tree = ET.parse('/notebooks/Thesis/1_Preprocessing/annotations_reformatted.xml')
    root = tree.getroot()

# Open video list
if NORMAL:
    video_list = os.listdir('/notebooks/data/Datasets/CPTAD/Videos_Normal/Test/')
else:
    video_list = pd.read_csv('/notebooks/Thesis/data_yolo/test_videos.txt', sep=" ", header=None)

cv2.namedWindow('YOLOv8 Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLOv8 Tracking', 1280, 720)

for i in range(len(video_list)):

    if NORMAL:
        video_id = i
        video_name = video_list[i]
    else:
        video_id = video_list.iloc[i, 0]
        video_name = video_list.iloc[i, 1]
        video = root.find(f"./video[@taskid='{video_id}']")

    # Open the video file
    if NORMAL:
        video_path = f"/notebooks/data/Datasets/CPTAD/Videos_Normal/Test/{video_name}"
    else:
        video_path = f"/notebooks/data/Datasets/CPTAD/Videos/{video_name}"
    print(video_path)

    cap = cv2.VideoCapture(video_path)

    if NORMAL:
        resultFile = f"/notebooks/Thesis/results/yolo_{MODEL}_norm/{video_name[:-4]}_{video_id}.csv"
    else:
        resultFile = f"/notebooks/Thesis/results/yolo_{MODEL}/{video_name[:-4]}_{video_id}.csv"
    # if os.path.isfile(resultFile):
    #     print(f"{resultFile} exists. skipping")
    #     continue
    
    resultsList = []
    # Loop through the video frames
    framenum = 0
    while cap.isOpened() and framenum < 1800: # only do a minute
        # Read a frame from the video
        framenum = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame
            results = model.predict(frame, conf=MIN_CONF)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            for result in results:
                for box in result.boxes:
                    # print(box)
                    # print([framenum]+box.cls.to("cpu").tolist()+box.conf.to("cpu").tolist()+box.xywh.to("cpu").tolist())
                    res = [framenum]+box.cls.cpu().tolist()+box.conf.cpu().tolist()+box.xywh.cpu().tolist()[0]
                    # b = box.xywh.to("cpu").tolist()
                    # res = res + b

                    # print(b)
                    print(res)
                    resultsList.append(res)

            if not NORMAL and ADD_ANNOTATIONS:
                # list of what classifications should be on patches
                boxes = []
                tracks = video.find('tracks')
                numtracks = int(video.find('numtracks').text)
                for i in range(numtracks):
                    framedata = tracks[i].find(f"./frame[@frame='{framenum}']")
                    el1 = tracks[i].attrib.get("Element1")
                    el2 = tracks[i].attrib.get("Element2")
                    if framedata is not None and el1 == "Car" and el2 == "Car":
                        boxes.append({'xtl':int(round(float(framedata.attrib.get('xtl')))),
                                    'ytl':int(round(float(framedata.attrib.get('ytl')))),
                                    'xbr':int(round(float(framedata.attrib.get('xbr')))),
                                    'ybr':int(round(float(framedata.attrib.get('ybr'))))})

            if not NORMAL and ADD_ANNOTATIONS:
                for box in boxes:
                    cv2.rectangle(annotated_frame, (box['xtl'],box['ytl']), (box['xbr'], box['ybr']), (255,255,0), 2)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # resultsList = resultsList.cpu()

    if len(resultsList) > 0:
        resultsList = np.array(resultsList).reshape(-1, len(resultsList[0]))
    resultsList = pd.DataFrame(resultsList, columns=['frame', 'cls', 'conf', 'x', 'y', 'w', 'h'])
    resultsList.to_csv(resultFile, index=False)

    # Release the video capture object and close the display window
    cap.release()

cv2.destroyAllWindows()