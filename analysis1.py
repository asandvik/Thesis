
import xml.etree.ElementTree as ET
import matplotlib.pylab as plt
import numpy as np

tree = ET.parse('annotations5_timelines.xml')
videos = tree.getroot()

statuses = {'accepted':0, 'rejected':0}
categories = {'before':0, 'ongoing':0, 'occluded':0, 'after':0}

durations = []
resolutions = {}
elements = {}

before = []
ongoing_occluded_after = []

before_ongoing_occluded = []
after = []

for video in videos.iter('video'):

    # Accepted vs Rejected
    status = video.find('status').text
    statuses[status] += 1

    if status == 'accepted':
        durations.append(float(video.find('length').text) / 30.0)

        width = video.find('width').text
        height = video.find('height').text
        res = (int(width), int(height))
        if res not in resolutions:
            label = width + 'x' + height
            resolutions[res] = {'label':label, 'count':1}
        else:
            resolutions[res]['count'] += 1
        
        for track in video.find('tracks').iter('track'):
            el1 = track.get('Element1')
            el2 = track.get('Element2')
            els = '/'.join(sorted([el1, el2]))
            if els not in elements:
                elements[els] = 1
            else:
                elements[els] += 1

        # Number of crash frames, non-crash frames, etc.
        before.append(0)
        ongoing_occluded_after.append(0)
        after.append(0)
        before_ongoing_occluded.append(0)
        totalframes = 0
        for interval in video.find('timeline').iter('interval'):
            nframes = int(interval.find('end').text) - int(interval.find('start').text) + 1 # add 1 -> same start and end is one frame
            totalframes += nframes
            category = interval.find('category').text
            categories[category] += nframes

            if category == 'before':
                before[-1] += nframes
            else:
                ongoing_occluded_after[-1] += nframes

            if category == 'after':
                after[-1] += nframes
            else:
                before_ongoing_occluded[-1] += nframes
    

fig1, ax1 = plt.subplots()
bar_container = ax1.bar(list(statuses.keys()), list(statuses.values()))
ax1.set(title='CADP Segment Status', ylabel='count')
ax1.bar_label(bar_container)

fig2, ax2 = plt.subplots()
bar_container = ax2.bar(list(categories.keys()), list(categories.values()))
ax2.set(title='Frame Category Counts', ylabel='count')
ax2.bar_label(bar_container)

fig3, ax3 = plt.subplots()
bins = range(0,92)
_, _, bars = ax3.hist(np.clip(durations, bins[0], bins[-1]), bins=bins)
ax3.set(title='Video Durations', xlabel='time (s)', ylabel='count')
ax3.bar_label(bars)
ticks = np.arange(0,91,5)
ax3.set_xticks(ticks)
xlabels = ticks.astype(str)
xlabels[-1] += '+'
ax3.set_xticklabels(xlabels)

fig4, ax4 = plt.subplots()
sorteddict = sorted(resolutions.items())
labels = [None] * len(sorteddict)
counts = [None] * len(sorteddict)
for i in range(len(sorteddict)):
    labels[i] = sorteddict[i][1]['label']
    counts[i] = sorteddict[i][1]['count']
bars = ax4.barh(labels, counts)
ax4.set(title='Video Resolutions', xlabel='count')
ax4.bar_label(bars)

fig5, ax5 = plt.subplots()
sorteddict = {k: v for k, v in sorted(elements.items(), key=lambda item: item[1])}
bars = ax5.barh(list(sorteddict.keys()), list(sorteddict.values()))
ax5.set(title='Crash Elements', xlabel='count')
ax5.bar_label(bars)

fig6, ax6 = plt.subplots()
ax6.scatter(before, ongoing_occluded_after, marker='.')
ax6.set(title='Crash Elements', xlabel='Frames before impact', ylabel='Frames after impact')
ax6.set_aspect('equal')

fig7, ax7 = plt.subplots()
ax7.scatter(before_ongoing_occluded, after, marker='.')
ax7.set(title='Crash Elements', xlabel='Frames before crash is over', ylabel='Frames after crash is over')
ax7.set_aspect('equal')

plt.show()