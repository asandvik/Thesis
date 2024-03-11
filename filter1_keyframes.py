"""
This script removes all non-keyframe boxes from the
CVAT's raw annotation file, resulting in a much
smaller file size without loss of relevent 
information.
"""
import xml.etree.ElementTree as ET

tree = ET.parse('annotations.xml')
root = tree.getroot()

for track in root.iter('track'):
    i = 0
    while i < len(track):
        if (track[i].attrib.get('keyframe') == "0"):
            track.remove(track[i])
        else:
            i += 1

tree.write('annotations_keyframes.xml')
