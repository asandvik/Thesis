import os
from time import sleep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

MODEL = 'cc3_bestAccuracy'
tree = ET.parse('/notebooks/Thesis/1_Preprocessing/annotations_reformatted.xml')
root = tree.getroot()

resultfiles = []
filelist = os.listdir(f'/notebooks/Thesis/results/{MODEL}')
for file in filelist:
    if file.endswith('.csv'):
        resultfiles.append(file)


for j in range(len(resultfiles)):

    filename = resultfiles[j]
    videoid = filename[-10:-4]

    print(filename)
    print(videoid)

    videoinfo = root.find(f"./video[@taskid='{videoid}']")
    tracks = videoinfo.find('tracks')

    impactframes = []
    for track in tracks:
        impactframes.append(int(track[0].attrib.get("frame")))

    results = pd.read_csv(f'/notebooks/Thesis/results/{MODEL}/{filename}')

    framenum = []
    numpatches = []
    numpositive = []
    prevpatchnum = -1
    j = -1

    for i in range(len(results)):
        currpatchnum = results.iloc[i, 0]
        currpatchres = results.iloc[i, 3]

        if currpatchnum != prevpatchnum:
            j += 1
            framenum.append(currpatchnum)
            numpatches.append(1)
            numpositive.append(currpatchres)
        else:
            numpatches[j] += 1
            numpositive[j] += currpatchres

        prevpatchnum = currpatchnum

    percentpositive = []
    for i in range(len(numpositive)):
        percentpositive.append(numpositive[i] / numpatches[i])

    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.vlines(x = impactframes, ymin = 0, ymax = max(numpositive), colors = 'red')
    ax1.plot(framenum, numpositive)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.vlines(x = impactframes, ymin = 0, ymax = max(percentpositive), colors = 'red')
    ax2.plot(framenum, percentpositive)

    fig1.savefig(f'/notebooks/Thesis/results/{MODEL}/{filename[0:-11]}_tot.png')
    fig2.savefig(f'/notebooks/Thesis/results/{MODEL}/{filename[0:-11]}_per.png')