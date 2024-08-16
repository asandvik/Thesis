import os
import argparse
import torch
import pandas as pd
import numpy as np
import seaborn as sn
import xml.etree.ElementTree as ET
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision import transforms as t
from torchvision.models.video.resnet import r2plus1d_18, R2Plus1D_18_Weights
# from torchvision.models.video.swin_transformer import swin3d_b Swin3D_B_Weights
from CPTADDataset3 import CPTADDataset3
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from transforms import ConvertBCHWtoCBHW
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--learn', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--batch', type=int, default=8, help='batch size')
parser.add_argument('--workers', type=int, default=4, help='number of dataloader workers')
parser.add_argument('--check', type=int, default=10, help='how many batches before sub-epoch reporting')
parser.add_argument('--epochs', type=int, default=30, help='maximum number of epochs')
parser.add_argument('--chkpnt', type=int, default=5, help='number of epochs before checkpointing')
parser.add_argument('--nframes', type=int, default=16, help='number of frames in sample')
parser.add_argument('--anno', type=str, default='sampleregions', help='annotation files directory')
parser.add_argument('--model', type=str, default='cnn', const='cnn', nargs='?', 
                    choices=['cnn','swin'], help='R(2+1)D or Swin Transformer')
parser.add_argument('--loader', type=str, default='regions', const='regions', nargs='?',
                    choices=['intervals','windows','regions'], help='data loader sampling technique')
parser.add_argument('--loss', type=str, default='BCE', const='BCE', nargs='?',
                    choices=['CE','BCE'], help='loss function')
parser.add_argument('tag', type=str, help='unique tag for this experiment')

args = parser.parse_args()
tag = args.tag

RUNS_DIR = '/notebooks/Thesis/runs/' + tag + '/'
MODEL_DIR = '/notebooks/Thesis/models/'

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
# for param in model.parameters(): # use as fixed feature extractor
#     param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model = model.to(device)

# criterion = torch.nn.BCEWithLogitsLoss()
criterion = torch.nn.CrossEntropyLoss()
    
optimizer = torch.optim.SGD(model.parameters(), lr=args.learn, momentum=args.momentum)

# https://pytorch.org/vision/0.8/models.html#video-classification
mean = [0.43216, 0.394666, 0.37645]
std = [0.22803, 0.22145, 0.216989]

# https://github.com/pytorch/vision/blob/main/references/video_classification/presets.py
# video_transform_train = transforms.Compose([
#                                 t.ConvertImageDtype(torch.float32),
#                                 t.RandomHorizontalFlip(),
#                                 t.Normalize(mean=mean,std=std),
#                                 ConvertBCHWtoCBHW()
# ])
                               
                                
training_data = CPTADDataset3("9DRFJxKHc6g06.mp4",
                             "../data/Datasets/CPTAD/Videos/",
                             "/notebooks/Thesis/1_Preprocessing/annotations_reformatted.xml",
                             nframes=16,
                             )

train_dataloader = DataLoader(training_data, batch_size=args.batch, num_workers=args.workers)
# valid_dataloader = DataLoader(valid_data, batch_size=args.batch, num_workers=args.workers)
# test_dataloader = DataLoader(test_data, batch_size=args.batch, num_workers=args.workers)

print(f" - Number of samples : {len(train_dataloader.dataset)} | Number of batches: {len(train_dataloader)}")

writer = SummaryWriter(RUNS_DIR)

for epoch in range(args.epochs):
    print('Epoch-{0} lr: {1}'.format(epoch+1, optimizer.param_groups[0]['lr']))

    # Training one epoch
    model.train(True)
    train_loss = 0
    train_true = []
    train_pred = []
    total_samples = 0
    
    for i, data in enumerate(train_dataloader, 0):

        inputs, labels = data
        inputs, labels = inputs.float().to(device), labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        train_true.extend(labels.data.cpu().numpy())
        train_pred.extend(predicted.data.cpu().numpy())

        # total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        # print(outputs)
        # print(labels)
        # print(predicted)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item() # average batch loss
        train_loss += batch_loss*inputs.size(0) # total loss for batch
            
    # End of Epoch
    
    print("total samples:", total_samples)

    train_ave_loss = train_loss/len(train_dataloader.dataset)

    train_f1 = f1_score(train_true, train_pred)

    train_precision = precision_score(train_true, train_pred)

    train_recall = recall_score(train_true, train_pred)

    train_accuracy = accuracy_score(train_true, train_pred)
    
    writer.add_scalar(tag + '/Training F1 Score',
                        train_f1,
                        epoch+1)
    
    writer.add_scalar(tag + '/Training Loss',
                    train_ave_loss,
                    epoch+1)

    writer.add_scalar(tag + '/Training Accuracy',
                train_accuracy,
                epoch+1)

    writer.add_scalar(tag + '/Training Recall',
                train_recall,
                epoch+1)
    
    writer.add_scalar(tag + '/Training Precision',
                    train_precision,
                    epoch+1)

    writer.flush()

    # checkpoint
    if (epoch+1) % args.chkpnt == 0:
        model_path = '{}{}_checkpoint'.format(MODEL_DIR,tag)
        print("Checkpointing model. Epoch: {}".format(epoch+1))
        torch.save(model.state_dict(), model_path)
