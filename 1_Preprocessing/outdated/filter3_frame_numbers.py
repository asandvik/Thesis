"""
This script calculates frame numbers within every
track so that they correspond to their frame number
in the videos. It is applied after filter2.
"""
import xml.etree.ElementTree as ET

tree = ET.parse('annotations_state_changes.xml')
root = tree.getroot()

for track in root.iter('track'):
    i = 0

    task_id = track.attrib.get('task_id')

    offset = 0
    for task in root[1][0][5].iter('task'):
        if task_id == task[0].text: # id
            break
        else:
            offset += int(task[2].text) # size

    for box in track.iter('box'):
        framenum = int(box.attrib.get('frame')) - offset
        box.attrib.update({'frame':str(framenum)})
        
tree.write('annotations3_filtered.xml')
