"""
This script 
"""
import xml.etree.ElementTree as ET
import copy

tree = ET.parse('annotations.xml')
root = tree.getroot()

# Adjust frame numbers so that they correspond to their
# frame number in the video.
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

# Reformat annotations
videos = ET.Element('videos')

in_tracks = root.findall('track')
num_tracks = len(in_tracks)
i = 0

for task in root[1][0][5].iter('task'):
    video = ET.SubElement(videos, 'video')

    joburl = task.find('segments').find('segment').find('url').text

    video.set('taskid', task.find('id').text)
    video.set('jobid', joburl[len(joburl)-6:])
    
    name = ET.SubElement(video, 'name')
    name.text = task.find('name').text
    length = ET.SubElement(video, 'length')
    length.text = task.find('size').text
    width = ET.SubElement(video, 'width')
    width.text = task.find('original_size').find('width').text
    height = ET.SubElement(video, 'height')
    height.text = task.find('original_size').find('height').text
    numtracks = ET.SubElement(video, 'numtracks')
    numtracks.text = '0'
    crashstart = ET.SubElement(video, 'crashstart')
    crashstart.text = '0'
    crashsettled = ET.SubElement(video, 'crashsettled')
    crashsettled.text = '0'
    status = ET.SubElement(video, 'status')
    status.text = 'rejected'

    tracks = ET.SubElement(video, 'tracks')
    timeline = ET.SubElement(video, 'timeline')

    while (i < num_tracks and in_tracks[i].attrib.get('task_id') == video.attrib.get('taskid')):
        status.text = 'accepted'
        numtracks.text = str(int(numtracks.text) + 1);
        track = copy.deepcopy(in_tracks[i])
        track.attrib.__delitem__('source')
        track.attrib.__delitem__('subset')
        track.set('Element1', track.find('box')[0].text)
        track.set('Element2', track.find('box')[1].text)
        for box in track.iter('box'):
            box.attrib.__delitem__('z_order')
            box.remove(box[1])
            box.remove(box[0])
            box.tag = 'frame'
        tracks.append(track)
        # ET.SubElement(tracks, track.tag, track.attrib)
        i += 1

    # Determine start and end/settle frames of crash
    if status.text == 'accepted':
        tracklist = tracks.findall('track')
        crashstart.text = tracklist[0][0].attrib.get('frame')
        settlelist = []
        for track in tracklist:
            framelist = track.findall('frame')
            nframes = len(framelist)
            track.set('start', framelist[0].attrib.get('frame'))
            currbbox = [framelist[nframes-1].attrib.get('xtl'),
                        framelist[nframes-1].attrib.get('ytl'),
                        framelist[nframes-1].attrib.get('xbr'),
                        framelist[nframes-1].attrib.get('ybr')]
            for j in range(nframes-2, -1, -1):
                nextbbox = [framelist[j].attrib.get('xtl'),
                            framelist[j].attrib.get('ytl'),
                            framelist[j].attrib.get('xbr'),
                            framelist[j].attrib.get('ybr')]
                if nextbbox != currbbox:
                    crashsettle = framelist[j+1].attrib.get('frame')
                    break
                elif j == 0:
                    crashsettle = framelist[j].attrib.get('frame')
            if nframes == 1:
                crashsettle = framelist[0].attrib.get('frame')
            track.set('settled', crashsettle)
            track.set('end', framelist[-1].attrib.get('frame'))
            settlelist.append(int(crashsettle))
        crashsettled.text = str(max(settlelist))

tree = ET.ElementTree(videos)
ET.indent(tree)

tree.write('annotations_reformatted.xml')
print("done")