"""
This script calculates 
"""
import xml.etree.ElementTree as ET
import copy

in_tree = ET.parse('annotations3_filtered.xml')
in_root = in_tree.getroot()

videos = ET.Element('videos')

# videos = copy.deepcopy(in_root[1][0][5])

for task in in_root[1][0][5].iter('task'):
    video = ET.SubElement(videos, 'video')
    name = ET.SubElement(video, 'name')
    taskid = ET.SubElement(video, 'taskid')
    joburl = ET.SubElement(video, 'jobid')
    nframes = ET.SubElement(video, 'nframes')

    name.text = task.find('name').text
    taskid.text = task.find('id').text
    nframes.text = task.find('size').text


# root = ET.Element('root')

# person = ET.SubElement(root, 'person')
# name = ET.SubElement(person, 'name')
# age = ET.SubElement(person, 'age')

# name.text = 'John Doe'
# age.text = '30'

tree = ET.ElementTree(videos)
ET.indent(tree)
tree.write('annotations4_timelines.xml')