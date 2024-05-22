import xml.etree.ElementTree as ET
import matplotlib.pylab as plt
import numpy as np
import random
import csv

# tree = ET.parse('annotations5_timelines.xml')
tree = ET.parse('annotations5_timelines.xml')
videos = tree.getroot()

id_list = []
for video in videos.iter('video'):
    status = video.find('status').text
    width = video.find('width').text
    if status == 'accepted' and width == '1280':
        id_list.append(video.get('taskid'))

random.Random(40).shuffle(id_list)

i = 0
num_vids = len(id_list)

def write_anno(writer, video):
    name = video.find('name').text
    tl = video.find('timeline')
    if len(tl) < 2: return # ignore no before-crash
    impactframe = int(tl[1][0].text)
    nocrashframe = (max(0, impactframe - 10))
    crashframe = str(impactframe + 10)
    writer.writerow([name, nocrashframe, '0'])
    writer.writerow([name, crashframe, '1'])

with open('anno_train.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(0, int(0.7*num_vids)):
        video = videos.find(f".//video[@taskid='{id_list[i]}']")
        write_anno(writer, video)

with open('anno_valid.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(int(0.7*num_vids), int(0.85*num_vids)):
        video = videos.find(f".//video[@taskid='{id_list[i]}']")
        write_anno(writer, video)

with open('anno_test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(int(0.85*num_vids), num_vids):
        video = videos.find(f".//video[@taskid='{id_list[i]}']")
        write_anno(writer, video)
