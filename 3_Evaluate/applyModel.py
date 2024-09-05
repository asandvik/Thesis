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

MODEL = 'aug6_bestAccuracy'
VIDEO_DIR = '/notebooks/data/Datasets/CPTAD/Videos/'
NORM_VIDEO_DIR = '/notebooks/data/Datasets/CPTAD/Videos_Normal/Test/'
NFRAMES_MD = 5
NFRAMES_01 = 48
PATCH_SIZE = 112
STRIDE = 56 # 112, 84, 56, 28
BIN_THRESH = 35 # grayscale to binary threshold. max 255
MOVE_THRESH = 200 # min number of pixels that need to be lit up in the binary movement image. max 112x112
MAX_BATCH = 20

try:
    os.mkdir(f'/notebooks/Thesis/results/{MODEL}')
except FileExistsError:
    pass

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

def classFromOverlap(tlx, tly, boxes):
    ret = 0
    bboxA = [tlx, tly, tlx+112, tly+112]
    maxoverlap = 0
    for box in boxes:
        bboxB = [box['xtl'], box['ytl'], box['xbr'], box['ybr']]
        maxoverlap = max(maxoverlap, overlap(bboxA, bboxB))

    if maxoverlap >= 0.5:
        ret = 1
    return ret    

tree = ET.parse('/notebooks/Thesis/1_Preprocessing/annotations_reformatted.xml')
root = tree.getroot()

vidlist = pd.read_csv('/notebooks/Thesis/annotations/carcar/anno_test.csv')
normvidlist = os.listdir(NORM_VIDEO_DIR)

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

cv2.namedWindow('raw', cv2.WINDOW_NORMAL)
cv2.resizeWindow('raw', 1280, 720)

for j in range(len(vidlist) + len(normvidlist)):

    hascrash = j < len(vidlist) # boolean to distinuish crash videos from normal

    if hascrash:
        video_id = vidlist.iloc[j, 0]
        video_name = vidlist.iloc[j, 1]
        # track_idx = vidlist.iloc[j, 2]

        # untested. skip if this video is same as previous. possible with multiple impacts/tracks
        if video_id == vidlist.iloc[j-1, 0]:
            continue

        cap = cv2.VideoCapture(VIDEO_DIR + video_name)
    else:
        video_id = j-len(vidlist)
        video_name = normvidlist[video_id]
        cap = cv2.VideoCapture(NORM_VIDEO_DIR + video_name)
        
    resultFile = f"/notebooks/Thesis/results/{MODEL}/{video_name[:-4]}_{video_id}.csv"
    if os.path.isfile(resultFile):
        print(f"{resultFile} exists. skipping")
        continue

    print(video_name)
    
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

    vidwriter = cv2.VideoWriter(f'/notebooks/Thesis/results/{MODEL}/{video_name}', 
                    cv2.VideoWriter_fourcc(*'XVID'), 
                    fps,
                    (width, height))

    if hascrash:
        video = root.find(f"./video[@taskid='{video_id}']")
        crashstart = int(video.find('crashstart').text)
        crashsettled = int(video.find('crashsettled').text)
        framenum = max(crashstart-120, 0) # start at either beginning or 4 seconds before crash
        lastframe = crashsettled + 60
    else:
        framenum = 0
        lastframe = 300 + NFRAMES_01

    cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
    results = []
    while framenum < lastframe:
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

        if hascrash:
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
                    
            # print(boxes)

        # list of classifications on patches
        frames_tensor = torch.tensor(np.array(frames_01))
        frames_tensor = torch.permute(frames_tensor, (0, 3, 1, 2)) # to TCHW
        
        num_patches = len(patchesTL)
        patch_idx_offset = 0
        while patch_idx_offset < num_patches:
            num_batch = min(num_patches-patch_idx_offset, MAX_BATCH)
            patches_tensor = torch.empty(num_batch, 3, NFRAMES_01, 112, 112)
            for i in range(num_batch):
                tl = patchesTL[i+patch_idx_offset]
                patch = F.crop(frames_tensor, tl[1], tl[0], 112, 112)
                patch = F.normalize(patch, mean=[0.43216, 0.394666, 0.37645],std=[0.22803, 0.22145, 0.216989])
                patch = torch.permute(patch, (1, 0, 2, 3)) # to CTHW
                patches_tensor[i] = patch

            # print(patches_tensor.size())

            with torch.no_grad():
                output = model(patches_tensor.to(device))

            _, predictions = torch.max(output, 1)

            if hascrash:
                for box in boxes:
                    cv2.rectangle(frame, (box['xtl'],box['ytl']), (box['xbr'], box['ybr']), (255,255,0), 2)

            for i in range(num_batch):
                tlx = patchesTL[i+patch_idx_offset][0]
                tly = patchesTL[i+patch_idx_offset][1]
                if hascrash:
                    target = classFromOverlap(tlx, tly, boxes)
                else:
                    target = 0
                results.append([framenum, tlx, tly, target, predictions[i].item()])

                match predictions[i].item():
                    case 0:
                        color = (0, 255, 0) # green
                    case 1:
                        color = (0, 0, 255) # red
                    case _:
                        print("Bad prediction")
                        
                cv2.rectangle(frame, (tlx,tly), (tlx+PATCH_SIZE-1, tly+PATCH_SIZE-1), color, 2)

            patch_idx_offset += MAX_BATCH

        cv2.imshow('raw', frame)
        vidwriter.write(frame)
        # cv2.imshow('delta', delta)
        # cv2.imshow('thresh', thresh)

        framenum += 1

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    if len(results) > 0:
        results = np.array(results).reshape(-1, len(results[0]))
    results = pd.DataFrame(results, columns=['frame', 'xtl', 'ytl', 'target', 'prediction'])
    results.to_csv(resultFile, index=False)

    cap.release()
    vidwriter.release()

cv2.destroyAllWindows()