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
# DIR = ROOT_DIR + 'len' + str(WIN_LENGTH) + 'strd' + str(WIN_STRIDE) + 'lim' + str(VID_FRAME_LIMIT) + '/'
DIR = ROOT_DIR + 'intervals' + str(WIN_LENGTH) + '/'

label_dict = {'before':0,
              'after':0,
              'occluded':1, # was 0 for write_anno()
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

def write_anno_interval(writer, video, prefix, ignore_file):
    name = video.find('name').text
    tl = video.find('timeline')

    nintervals = len(tl)

    for i in range(nintervals):
        start = int(tl[i][0].text)
        end = int(tl[i][1].text)
        category = tl[i][2].text

        if end < WIN_LENGTH or (end - start) < 8: # exclude brief intervals
            log = "{}: excluding {} {} interval: {}-{}".format(prefix, name,category,start,end)
            print(log)
            ignore_file.write(log)
            ignore_file.write('\n')
            continue

        start = max(WIN_LENGTH-1, start) # ensure interval has enough prior frames
        writer.writerow([name, str(start), str(end), str(label_dict[category])])

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

train_file = open(DIR + 'anno_train.csv', 'w', newline='')
valid_file = open(DIR + 'anno_valid.csv', 'w', newline='')
test_file = open(DIR + 'anno_test.csv', 'w', newline='')

train_writer = csv.writer(train_file)
valid_writer = csv.writer(valid_file)
test_writer = csv.writer(test_file)

ignore_file = open(DIR + 'ignored.txt', 'w')

for i in range(0, num_vids):
    video = videos.find(f".//video[@taskid='{id_list[i]}']")
    if (i % 10 == 0):
        write_anno_interval(valid_writer, video, 'VAL', ignore_file)
    elif (i % 10 == 1):
        write_anno_interval(test_writer, video, 'TEST', ignore_file)
    else:
        write_anno_interval(train_writer, video, 'TRAIN', ignore_file)

train_file.close()
valid_file.close()
test_file.close()
ignore_file.close()

# with open(DIR + 'anno_train.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     for i in range(0, int(0.8*num_vids)):
#         video = videos.find(f".//video[@taskid='{id_list[i]}']")
#         write_anno(writer, video)

# with open(DIR + 'anno_valid.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     for i in range(int(0.8*num_vids), int(0.90*num_vids)):
#         video = videos.find(f".//video[@taskid='{id_list[i]}']")
#         write_anno(writer, video)

# with open(DIR + 'anno_test.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     for i in range(int(0.90*num_vids), num_vids):
#         video = videos.find(f".//video[@taskid='{id_list[i]}']")
#         write_anno(writer, video)
