import os
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision import transforms as t
from torchvision.models.video.resnet import r2plus1d_18, R2Plus1D_18_Weights
from CPTADDataset import CPTADDataset
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from transforms import ConvertBCHWtoCBHW

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT).to(device)

# https://pytorch.org/vision/0.8/models.html#video-classification
mean = [0.43216, 0.394666, 0.37645]
std = [0.22803, 0.22145, 0.216989]

# https://github.com/pytorch/vision/blob/main/references/video_classification/presets.py
frame_transform_train = transforms.Compose([
                                t.ConvertImageDtype(torch.float32),
                                # horizontal flip? Not sure if belongs in frame or video transform
                                t.Resize((128, 171)),
                                t.RandomCrop((112, 112)),
                                t.Normalize(mean=mean,std=std),
                                ConvertBCHWtoCBHW()
])
                               

frame_transform_eval = transforms.Compose([
                                t.ConvertImageDtype(torch.float32),
                                t.Resize((128, 171)),
                                t.CenterCrop((112, 112)),
                                t.Normalize(mean=mean,std=std),
                                ConvertBCHWtoCBHW()
])
                                
nframes = 16
training_data = CPTADDataset("anno_train.csv", 
                             "../data/Datasets/CPTAD/Videos/", 
                             nframes=nframes,
                             video_transform=frame_transform_train)

valid_data = CPTADDataset("anno_valid.csv",
                            "../../data/Datasets/CPTAD/Videos",
                            nframes=nframes,
                            video_transform=frame_transform_eval)
    
test_data = CPTADDataset("anno_test.csv", 
                            "../data/Datasets/CPTAD/Videos/", 
                            nframes=nframes,
                            video_transform=frame_transform_eval)
batch_size = 8
train_dataloader = DataLoader(training_data, batch_size=batch_size)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0
    last_loss = 0

    for i, data in enumerate(train_dataloader):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # print("{} {}".format(inputs.size(), labels.size()))

        optimizer.zero_grad()
        outputs = model(inputs)
        # print(labels)
        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 5 == 0:
            last_loss = running_loss / 1000
            print('  batch {} loss: {}'.format(i, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0

    return last_loss

timestamp = datetime.now().strftime('%Y%m%d_%H%M%s')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000

for epoch in range(EPOCHS):
    print('EPOCH {}'.format(epoch_number + 1))

    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0

    model.eval()

    with torch.no_grad():
        for i, vdata in enumerate(valid_dataloader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    writer.add_scalars('Testing vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_loss },
                        epoch_number + 1)
    writer.flush()

    if avg_vloss < best_vloss:
        best_vloss = avg_loss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

