import xml.etree.ElementTree as ET
import matplotlib.pylab as plt
import numpy as np
import random
import csv
import os
from math import floor

ROOT_DIR = '/notebooks/Thesis/annotations/'
DIR = ROOT_DIR + 'carcar/'

tree = ET.parse('annotations_reformatted.xml')
videos = tree.getroot()

# List of suitable videos to use
id_list = []
for video in videos.iter('video'):
    status = video.find('status').text
    numtracks = int(video.find('numtracks').text)
    if status == 'accepted':
        tracks = video.find('tracks')
        for track in tracks:
            el1 = track.attrib.get('Element1')
            el2 = track.attrib.get('Element2')
            if el1 == 'Car' and el2 == 'Car': # if video contains a car/car crash at all
                id_list.append(video.get('taskid')) 
                break


random.Random(40).shuffle(id_list)

i = 0
num_vids = len(id_list)

def add_to_csv(writer, taskid):
    video = videos.find(f".//video[@taskid='{taskid}']")
    name = video.find('name').text
    tracks = video.find('tracks')
    i = 0
    for track in tracks:
        el1 = track.attrib.get('Element1')
        el2 = track.attrib.get('Element2')
        end = int(track.attrib.get('end'))
        if end < 16: # skip impacts that end before 16 frames
            continue
        if el1 == 'Car' and el2 == 'Car':
            writer.writerow([taskid, name, i])
        i += 1
                
try:
    os.mkdir(DIR)
except FileExistsError:
    pass

train_file = open(DIR + 'anno_train.csv', 'w', newline='')
valid_file = open(DIR + 'anno_valid.csv', 'w', newline='')
test_file = open(DIR + 'anno_test.csv', 'w', newline='')

train_writer = csv.writer(train_file)
valid_writer = csv.writer(valid_file)
test_writer = csv.writer(test_file)

for i in range(0, num_vids):
    if (i % 10 == 0):
        add_to_csv(valid_writer, id_list[i])
    elif (i % 10 == 1):
        add_to_csv(test_writer, id_list[i])
    else:
        add_to_csv(train_writer, id_list[i])

train_file.close()
valid_file.close()
test_file.close()