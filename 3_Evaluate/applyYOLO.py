import os
import cv2
import pandas as pd
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("/notebooks/Thesis/2_Training/runs/detect/train2/weights/best.pt")

# Open video list
# video_list = pd.read_csv('/notebooks/Thesis/data_yolo/test_videos.txt', sep=" ", header=None)

normal_video_list = os.listdir('/notebooks/data/Datasets/CPTAD/Videos_Normal/Test/')

for i in range(len(normal_video_list)):
# for i in range(len(video_list)):

    video_name = normal_video_list[i]
    # video_id = video_list.iloc[i, 0]
    # video_name = video_list.iloc[i, 1]

    # Open the video file
    # video_path = f"/notebooks/data/Datasets/CPTAD/Videos/{video_name}"
    video_path = f"/notebooks/data/Datasets/CPTAD/Videos_Normal/Test/{video_name}"
    print(video_path)

    cap = cv2.VideoCapture(video_path)
    
    # Loop through the video frames
    print(cap.isOpened())
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame
            results = model.track(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()

cv2.destroyAllWindows()