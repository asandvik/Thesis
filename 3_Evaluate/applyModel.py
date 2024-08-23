import os
import cv2
import numpy as np
import pandas as pd
import collections
import torch
import torchvision.transforms.functional as F
from torchvision.models.video.resnet import r2plus1d_18
from torchvision import transforms as t
from time import sleep
import xml.etree.ElementTree as ET

MODEL = 'cc3_bestAccuracy'
VIDEO_DIR = '/notebooks/data/Datasets/CPTAD/Videos/'
NFRAMES_MD = 5
NFRAMES_01 = 16
PATCH_SIZE = 112
STRIDE = 56 # 112, 84, 56, 28
BIN_THRESH = 35 # grayscale to binary threshold. max 255
MOVE_THRESH = 200 # min number of pixels that need to be lit up in the binary movement image. max 112x112
MAX_BATCH = 64

try:
    os.mkdir(f'/notebooks/Thesis/results/{MODEL}')
except FileExistsError:
    pass

tree = ET.parse('/notebooks/Thesis/1_Preprocessing/annotations_reformatted.xml')
root = tree.getroot()

# video_name = random.choice(os.listdir(VIDEO_DIR))
# print(video_name)

# video_name = 'nsHYg3ENgk408.mp4'
# video_name = '9DRFJxKHc6g06.mp4' # good test video
# LK2OAH5ksKU01.mp4 # edge of frame

vidlist = pd.read_csv('/notebooks/Thesis/annotations/carcar/anno_test.csv')

device = ("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f"Using {device} device")

model = r2plus1d_18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model = model.to(device)
model.load_state_dict(torch.load(f'/notebooks/Thesis/models/{MODEL}'))
model.eval()
model.to(device)

for j in range(len(vidlist)):
    video_id = vidlist.iloc[j, 0]
    video_name = vidlist.iloc[j, 1]
    track_idx = vidlist.iloc[j, 2]

    # untested. skip if this video is same as previous. possible with multiple impacts/tracks
    if video_id == vidlist.iloc[j-1, 0]:
        continue

    print(video_name)

    cap = cv2.VideoCapture(VIDEO_DIR + video_name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    numAcross = width // PATCH_SIZE
    numDown = height // PATCH_SIZE

    remAcross = width % PATCH_SIZE
    remDown = height % PATCH_SIZE

    frames_md = collections.deque(maxlen=NFRAMES_MD)
    frames_01 = collections.deque(maxlen=NFRAMES_01)

    video = root.find(f"./video[@taskid='{video_id}']")
    crashstart = int(video.find('crashstart').text)
    crashsettled = int(video.find('crashsettled').text)

    vidwriter = cv2.VideoWriter(f'/notebooks/Thesis/results/{MODEL}/{video_name}', 
                        cv2.VideoWriter_fourcc(*'XVID'), 
                        fps,
                        (width, height))

    framenum = max(crashstart-120, 0) # start at either beginning or 4 seconds before crash
    cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
    results = []
    while framenum < crashsettled + 60:
        ret, frame = cap.read()
        
        # if no more frames
        if frame is None:
            break

        frames_01.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        frames_md.append(gray)

        # if buffer is not full
        if len(frames_01) is not NFRAMES_01:
            cv2.imshow('raw', frame)
            framenum += 1
            continue

        # movement binary image    
        delta = cv2.absdiff(frames_md[0], frames_md[-1])
        thresh = cv2.threshold(delta, BIN_THRESH, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=1)

        # list of patches w/ movement top left corner 
        patchesTL = []

        for y in range(int(remDown // 2), height - PATCH_SIZE, STRIDE):
            for x in range(int(remAcross // 2), width - PATCH_SIZE, STRIDE):

                threshPatch = thresh[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                if sum(sum(threshPatch)) > MOVE_THRESH:
                    # cv2.rectangle(frame, (x,y), (x+PATCH_SIZE-1, y+PATCH_SIZE-1), (255,255,255), 1)
                    patchesTL.append((x,y))

        # list of classifications on patches
        frames_tensor = torch.tensor(np.array(frames_01))
        frames_tensor = torch.permute(frames_tensor, (0, 3, 1, 2)) # to TCHW
        
        num_patches = len(patchesTL)
        patch_idx_offset = 0
        while patch_idx_offset < num_patches:
            num_batch = min(num_patches-patch_idx_offset, MAX_BATCH)
            patches_tensor = torch.empty(num_batch, 3, 16, 112, 112)
            for i in range(num_batch):
                tl = patchesTL[i+patch_idx_offset]
                patch = F.crop(frames_tensor, tl[1], tl[0], 112, 112)
                patch = F.normalize(patch, mean=[0.43216, 0.394666, 0.37645],std=[0.22803, 0.22145, 0.216989])
                patch = torch.permute(patch, (1, 0, 2, 3)) # to CTHW
                patches_tensor[i] = patch

            print(patches_tensor.size())

            with torch.no_grad():
                output = model(patches_tensor.to(device))

            _, predictions = torch.max(output, 1)

            for i in range(num_batch):
                tlx = patchesTL[i+patch_idx_offset][0]
                tly = patchesTL[i+patch_idx_offset][1]
                results.append([framenum, tlx, tly, predictions[i].item()])

                match predictions[i].item():
                    case 0:
                        color = (0, 255, 0) # green
                    case 1:
                        color = (0, 0, 255) # red
                    case _:
                        print("Bad prediction")
                        
                cv2.rectangle(frame, (tlx,tly), (tlx+PATCH_SIZE-1, tly+PATCH_SIZE-1), color, 1)

            patch_idx_offset += MAX_BATCH

        cv2.imshow('raw', frame)
        vidwriter.write(frame)
        cv2.imshow('delta', delta)
        cv2.imshow('thresh', thresh)

        framenum += 1

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    results = np.array(results).reshape(-1, len(results[0]))
    results = pd.DataFrame(results)
    results.to_csv(f"/notebooks/Thesis/results/{MODEL}/{video_name[:-4]}_{video_id}.csv", index=False, header=False)

    cap.release()
    vidwriter.release()

cv2.destroyAllWindows()