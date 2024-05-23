import xml.etree.ElementTree as ET
import matplotlib.pylab as plt
import numpy as np
import random
import csv
import os

WIN_LENGTH = 16
WIN_STRIDE = 8
VID_FRAME_LIMIT = 600

ROOT_DIR = '/notebooks/Thesis/annotations/'
DIR = ROOT_DIR + 'len' + str(WIN_LENGTH) + 'strd' + str(WIN_STRIDE) + 'lim' + str(VID_FRAME_LIMIT) + '/'

label_dict = {'before':0,
              'after':0,
              'occluded':0,
              'ongoing':1}

tree = ET.parse('annotations5_timelines.xml')
videos = tree.getroot()

# List of suitable videos to use
id_list = []
for video in videos.iter('video'):
    status = video.find('status').text
    width = video.find('width').text
    length = int(video.find('length').text)
    if status == 'accepted' and length < VID_FRAME_LIMIT:
        id_list.append(video.get('taskid'))

random.Random(40).shuffle(id_list)

i = 0
num_vids = len(id_list)

def write_anno(writer, video):
    name = video.find('name').text
    tl = video.find('timeline')
    duration = int(video.find('length').text)
    frame_num = WIN_LENGTH - 1
    label_counts = [0, 0]
    interval = 0
    interval_last_frame = int(tl[interval][1].text)
    while frame_num < duration:
        while frame_num > interval_last_frame:
            interval += 1
            interval_last_frame = int(tl[interval][1].text)
        category = tl[interval].find('category').text
        writer.writerow([name, str(frame_num), str(label_dict[category])])
        frame_num += WIN_STRIDE

os.mkdir(DIR)

with open(DIR + 'anno_train.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(0, int(0.8*num_vids)):
        video = videos.find(f".//video[@taskid='{id_list[i]}']")
        write_anno(writer, video)

with open(DIR + 'anno_valid.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(int(0.8*num_vids), int(0.90*num_vids)):
        video = videos.find(f".//video[@taskid='{id_list[i]}']")
        write_anno(writer, video)

with open(DIR + 'anno_test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(int(0.90*num_vids), num_vids):
        video = videos.find(f".//video[@taskid='{id_list[i]}']")
        write_anno(writer, video)
