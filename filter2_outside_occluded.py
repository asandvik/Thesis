"""
This script removes boxes where the "outside" and
"occluded" attributes are the same from the previous
box. It is applied after filter1
"""
import xml.etree.ElementTree as ET

tree = ET.parse('annotations_keyframes.xml')
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
        
tree.write('annotations2_state_changes.xml')
