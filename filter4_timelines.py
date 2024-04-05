"""
This script calculates 
"""
import xml.etree.ElementTree as ET
import copy

in_tree = ET.parse('annotations3_filtered.xml')
in_root = in_tree.getroot()

videos = ET.Element('videos')

# videos = copy.deepcopy(in_root[1][0][5])

in_tracks = in_root.findall('track')
num_tracks = len(in_tracks)
i = 0

for task in in_root[1][0][5].iter('task'):
    video = ET.SubElement(videos, 'video')

    joburl = task.find('segments').find('segment').find('url').text

    video.set('name', task.find('name').text)
    video.set('taskid', task.find('id').text)
    video.set('jobid', joburl[len(joburl)-6:])
    video.set('nframes', task.find('size').text)
    video.set('width', task.find('original_size').find('width').text)
    video.set('height', task.find('original_size').find('height').text)

    tracks = ET.SubElement(video, 'tracks')
    timeline = ET.SubElement(video, 'timeline')

    while (i < num_tracks and in_tracks[i].attrib.get('task_id') == video.attrib.get('taskid')):
        track = copy.deepcopy(in_tracks[i])
        track.attrib.__delitem__('source')
        track.attrib.__delitem__('subset')
        track.set('Element1', track.find('box')[0].text)
        track.set('Element2', track.find('box')[1].text)
        for box in track.iter('box'):
            box.attrib.__delitem__('xtl')
            box.attrib.__delitem__('ytl')
            box.attrib.__delitem__('xbr')
            box.attrib.__delitem__('ybr')
            box.attrib.__delitem__('z_order')
            box.remove(box[1])
            box.remove(box[0])
            box.tag = 'frame'
        tracks.append(track)
        # ET.SubElement(tracks, track.tag, track.attrib)
        i += 1

tree = ET.ElementTree(videos)
ET.indent(tree)
tree.write('annotations4_timelines.xml')