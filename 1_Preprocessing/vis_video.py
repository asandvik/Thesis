"""
This script 
"""

import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from mpl_toolkits.mplot3d import Axes3D
import cv2
from matplotlib.animation import FuncAnimation

def grab_frame(cap):
    ret,frame = cap.read()
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

tree = ET.parse('annotations_reformatted.xml')
root = tree.getroot()

index = 0

video = root[index]
filename = video.find('name').text
filepath = '/notebooks/data/Datasets/CPTAD/Videos/' + filename
cap = cv2.VideoCapture(filepath)

length = int(video.find('length').text)
width = int(video.find('width').text)
height = int(video.find('height').text)

fig = plt.figure(figsize=(16,9))

ax = fig.add_subplot(1,2,1,projection='3d')
ax.set_xlim(0, length)
ax.set_ylim(0, width)
ax.set_zlim(0, height)
ax.invert_zaxis()
ax.invert_xaxis()
ax.set_xlabel('frame')
ax.set_ylabel('width')
ax.set_zlabel('height')
ax.view_init(elev=0, azim=0, roll=0)

ax2 = fig.add_subplot(1,2,2)
im1 = ax2.imshow(grab_frame(cap))

framenum = 0

def update(i):
    im1.set_data(grab_frame(cap))
    framenum += 1

    frames = video.find('tracks').findall('track').findall(f"frame[@frame='{cf}']")
    for frame in frames:
        
        xtl = float(frame.attrib.get('xtl'))
        ytl = float(frame.attrib.get('ytl'))
        xbr = float(frame.attrib.get('xbr'))
        ybr = float(frame.attrib.get('ybr'))
    
        # alpha = 1 if framenum == target_frame else 0.5
        alpha = 0.5
        
        ax.plot(framenum, xtl, ytl, 'r.', alpha=alpha)
        ax.plot(framenum, xbr, ybr, 'g.', alpha=alpha)
        ax.plot(framenum, xtl, ybr, 'b.', alpha=alpha)
        ax.plot(framenum, xbr, ytl, 'y.', alpha=alpha)


# cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame-1)

ani = FuncAnimation(plt.gcf(), update, interval=200, cache_frame_data=False)

def close(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)

cid = plt.gcf().canvas.mpl_connect("key_press_event", close)

plt.show()