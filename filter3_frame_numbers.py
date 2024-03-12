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
    prev_out = None
    prev_occ = None
    while i < len(track):
        if (track[i].attrib.get('outside') == prev_out and
            track[i].attrib.get('occluded') == prev_occ):
            track.remove(track[i])
        else:
            prev_out = track[i].attrib.get('outside')
            prev_occ = track[i].attrib.get('occluded')
            i += 1
        
tree.write('annotations_filtered.xml')
