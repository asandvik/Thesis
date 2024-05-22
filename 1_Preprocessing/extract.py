import ffmpeg
import json
import os

video_folder = 'C://Users//Addison//Downloads//forth_investigation//forth_investigation'
output_folder = 'C://Users//Addison//OneDrive - Cal Poly//Thesis//Videos'

f = open('annotations.json')

data = json.load(f)

annotated_video_names = data.keys()

# print(video_names);

vid_names = os.listdir(video_folder)

i = 1
for video_name in annotated_video_names:
    annotations = data.get(video_name)
    file_name = video_folder + "//" + video_name
    print(file_name)
    j = 1
    for annotation in annotations:
        Start = annotation.get('keyframes')[0].get('frame')
        End = annotation.get('keyframes')[1].get('frame')
        # print(i, ". ", video_name, " ", start, " ", end)
        outfile = f"{output_folder}//{video_name.split(".")[0]}{j:02}.mp4"
        # outfile = "test.mp4"
        print(outfile)
        input_file = ffmpeg.input(file_name)
        pts = "PTS-STARTPTS"
        output_file = input_file.trim(start=Start,end=End).setpts(pts)
        output = ffmpeg.output(output_file, outfile, format="mp4")
        output.run()
        i += 1
        j += 1
        
f.close()