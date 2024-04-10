
"""
This script calculates 
"""
import xml.etree.ElementTree as ET
import copy

tree = ET.parse('annotations4_tracks.xml')
root = tree.getroot()

# in order of priority from low to high
categories = ['before-crash', 'after-crash', 'occluded-crash', 'ongoing-crash']

for video in root.iter('video'):

    tracks = video.find('tracks')
    timeline = video.find('timeline')

    length = int(video.find('length').text)
    frames = [0] * length

    for track in tracks:
        startframe = None
        endframe = None
        symbol = 0
        framelist = track.findall('frame')
        for j in range(0, len(framelist)):

            frame = framelist[j]

            startframe = int(frame.attrib.get('frame'))
            if j < len(framelist)-1:
                endframe = int(framelist[j+1].attrib.get('frame'))
            else:
                endframe = length

            if frame.attrib.get('outside') == '1':
                symbol = 1
            elif frame.attrib.get('occluded') == '1':
                symbol = 2
            else:
                symbol = 3

            for i in range(startframe,endframe):
                if frames[i] < symbol:
                    frames[i] = symbol       

    start = 0
    for i in range(length):
        if (i == length-1 or frames[i] != frames[i+1]):
            interval = ET.SubElement(timeline, 'interval')

            intstart = ET.SubElement(interval, 'start')
            intstart.text = str(start)
            intend = ET.SubElement(interval, 'end')
            intend.text = str(i)
            category = ET.SubElement(interval, 'category')
            category.text = categories[frames[i]]
            
            start = i+1

ET.indent(tree)
tree.write('annotations5_timelines.xml')