import cv2
import numpy as np
import pandas as pd
import collections
from time import sleep


VIDEO_DIR = '/notebooks/data/Datasets/CPTAD/Videos/'
NFRAMES_MD = 5
NFRAMES_01 = 16
PATCH_SIZE = 112
STRIDE = 112 # 112, 84, 56, 28
BIN_THRESH = 30 # grayscale to binary threshold. max 255
MOVE_THRESH = 200 # min number of pixels that need to be lit up in the binary movement image. max 112x112

# randvid = random.choice(os.listdir(VIDEO_DIR))
# print(randvid)

randvid = '9DRFJxKHc6g06.mp4' # good test video
cap = cv2.VideoCapture(VIDEO_DIR + randvid)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

numAcross = width // PATCH_SIZE
numDown = height // PATCH_SIZE

remAcross = width % PATCH_SIZE
remDown = height % PATCH_SIZE

frames_md = collections.deque(maxlen=NFRAMES_MD)

framenum = 0
patchList = []
while (True):
    ret, frame = cap.read()
    
    # if no more frames
    if frame is None:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (17, 17), 0)

    frames_md.append(gray)

    # if buffer is not full
    if framenum < 15:
        cv2.imshow('raw', frame)
        framenum += 1
        continue

    # movement binary image    
    delta = cv2.absdiff(frames_md[0], frames_md[-1])
    thresh = cv2.threshold(delta, BIN_THRESH, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=1)

    # list of patches w/ movement top left corner 

    for y in range(int(remDown // 2), height - PATCH_SIZE, STRIDE):
        for x in range(int(remAcross // 2), width - PATCH_SIZE, STRIDE):

            threshPatch = thresh[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            if sum(sum(threshPatch)) > MOVE_THRESH:
                cv2.rectangle(frame, (x,y), (x+PATCH_SIZE-1, y+PATCH_SIZE-1), (255,255,255), 1)
                patchList.append([randvid, framenum, x, y])

    cv2.imshow('raw', frame)

    framenum += 1

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

patchList = np.array(patchList).reshape(-1, len(patchList[0]))
patchList = pd.DataFrame(patchList)
patchList.to_csv(f"/notebooks/Thesis/annotations/onevideo/{randvid[:-4]}.csv", index=False, header=False)

cap.release()
cv2.destroyAllWindows()