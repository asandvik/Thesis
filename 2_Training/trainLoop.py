import os
import argparse
import torch
import pandas as pd
import numpy as np
import seaborn as sn
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision import transforms as t
from torchvision.models.video.resnet import r2plus1d_18, R2Plus1D_18_Weights
# from torchvision.models.video.swin_transformer import swin3d_b Swin3D_B_Weights
from CPTADDataset import CPTADDataset
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from transforms import ConvertBCHWtoCBHW
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--learn', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--batch', type=int, default=16, help='batch size')
parser.add_argument('--workers', type=int, default=4, help='number of dataloader workers')
parser.add_argument('--check', type=int, default=10, help='how many batches before sub-epoch reporting')
parser.add_argument('--epochs', type=int, default=10, help='maximum number of epochs')
parser.add_argument('--chkpnt', type=int, default=5, help='number of epochs before checkpointing')
parser.add_argument('--nframes', type=int, default=16, help='number of frames in sample')
parser.add_argument('--anno', type=str, default='intervals16', help='annotation files directory')
parser.add_argument('--model', type=str, default='cnn', const='cnn', nargs='?', 
                    choices=['cnn','swin'], help='R(2+1)D or Swin Transformer')
parser.add_argument('--loader', type=str, default='intervals', const='intervals', nargs='?',
                    choices=['intervals','windows'], help='data loader sampling technique')
parser.add_argument('--loss', type=str, default='CE', const='CE', nargs='?',
                    choices=['CE','BCE'], help='loss function')
parser.add_argument('tag', type=str, help='unique tag for this experiment')

args = parser.parse_args()
tag = args.tag

# ANNO = 'len16strd8lim600'

ANNO_DIR = '/notebooks/Thesis/annotations/' + args.anno + '/'
RUNS_DIR = '/notebooks/Thesis/runs/' + tag + '/'
MODEL_DIR = '/notebooks/Thesis/models/'

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

if args.model == 'cnn':
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
    # for param in model.parameters(): # use as fixed feature extractor
    #     param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    model = model.to(device)
    
elif args.model == 'swin':
    model = swin3d_b(weights=Swin3D_B_Weights.DEFAULT).to(device)
else:
    error('invalid model. argparse should have caught.')

if args.loss == 'CE':
    criterion = torch.nn.BCEWithLogitsLoss()
elif args.loss == 'BCE':
    criterion = torch.nn.CrossEntropyLoss()
else:
    error('invalid loss function. argparse should have caught.')
    
optimizer = torch.optim.SGD(model.parameters(), lr=args.learn, momentum=args.momentum)

# https://pytorch.org/vision/0.8/models.html#video-classification
mean = [0.43216, 0.394666, 0.37645]
std = [0.22803, 0.22145, 0.216989]

# TODO: transforms for swin

# https://github.com/pytorch/vision/blob/main/references/video_classification/presets.py
video_transform_train = transforms.Compose([
                                t.ConvertImageDtype(torch.float32),
                                # horizontal flip? Not sure if belongs in frame or video transform
                                t.Resize((128, 171)),
                                t.RandomCrop((112, 112)),
                                t.Normalize(mean=mean,std=std),
                                ConvertBCHWtoCBHW()
])
                               

video_transform_eval = transforms.Compose([
                                t.ConvertImageDtype(torch.float32),
                                t.Resize((128, 171)),
                                t.CenterCrop((112, 112)),
                                t.Normalize(mean=mean,std=std),
                                ConvertBCHWtoCBHW()
])
                                
training_data = CPTADDataset(ANNO_DIR + "anno_train.csv", 
                             "../data/Datasets/CPTAD/Videos/", 
                             nframes=args.nframes,
                             video_transform=video_transform_train,
                             sample_tech=args.loader)

valid_data = CPTADDataset(ANNO_DIR + "anno_valid.csv",
                            "../../data/Datasets/CPTAD/Videos",
                            nframes=args.nframes,
                            video_transform=video_transform_eval,
                            sample_tech=args.loader)

# TODO: change sample_tech to window
test_data = CPTADDataset(ANNO_DIR + "anno_test.csv", 
                            "../data/Datasets/CPTAD/Videos/", 
                            nframes=args.nframes,
                            video_transform=video_transform_eval,
                            sample_tech=args.loader)

print("NUM SAMPLES train:{} validation:{} test:{}".format(len(training_data), len(valid_data), len(test_data)))

train_dataloader = DataLoader(training_data, batch_size=args.batch, num_workers=args.workers)
valid_dataloader = DataLoader(valid_data, batch_size=args.batch, num_workers=args.workers)
test_dataloader = DataLoader(test_data, batch_size=args.batch, num_workers=args.workers)

writer = SummaryWriter(RUNS_DIR)

val_accuracy_best = 0
val_recall_best = 0

for epoch in range(args.epochs):
    print('Epoch-{0} lr: {1}'.format(epoch+1, optimizer.param_groups[0]['lr']))

    # Training one epoch
    model.train(True)
    train_loss = 0
    train_true = []
    train_pred = []
    
    for i, data in enumerate(train_dataloader):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        train_true.extend(labels.data.cpu().numpy())
        train_pred.extend(predicted.data.cpu().numpy())

        # total_correct += (predicted == labels).sum().item()
        # total_samples += labels.size(0)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item() # average batch loss
        train_loss += batch_loss*inputs.size(0) # total loss for batch

        # if i % args.check == args.check-1:
        #     # print('  batch {} loss: {}'.format(i, batch_loss))
        #     tb_x = epoch * len(train_dataloader) + i + 1
        #     writer.add_scalars(tag + '/BatchLoss', {'Epoch_'+str(epoch+1):batch_loss}, tb_x)
        #     writer.flush()
            
    # End of epoch
    
    # Validating after training epoch
    model.eval()
    # vrunning_loss = 0.0
    # vnum_correct = 0
    # vtotal_samples = 0
    val_loss = 0
    val_true = []
    val_pred = []
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
    
            val_true.extend(labels.data.cpu().numpy())
            val_pred.extend(predicted.data.cpu().numpy())
    
            loss = criterion(outputs, labels)
            val_loss += loss.item()*inputs.size(0)

    train_ave_loss = train_loss/len(train_dataloader.dataset)
    val_ave_loss = val_loss/len(valid_dataloader.dataset)

    train_f1 = f1_score(train_true, train_pred)
    val_f1 = f1_score(val_true, val_pred)

    train_precision = precision_score(train_true, train_pred)
    val_precision = precision_score(val_true, val_pred)

    train_recall = recall_score(train_true, train_pred)
    val_recall = recall_score(val_true, val_pred)

    train_accuracy = accuracy_score(train_true, train_pred)
    val_accuracy = accuracy_score(val_true, val_pred)

    # print('F1 SCORE train {} val {}'.format(train_f1,val_f1))

    # constant for classes
    # classes = ('No Crash', 'Crash')
    
    # confusion matrix
    val_cf_matrix = confusion_matrix(val_true, val_pred)
    # df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
    #                      columns=[i for i in classes])
    # plt.figure(figsize=(12, 7))
    # cm_fig = sn.heatmap(df_cm, annot=True).get_figure()
    
    # writer.add_figure(tag + '/Validation Confusion matrix', cm_fig, epoch)

    train_cf_matrix = confusion_matrix(train_true, train_pred)
    # df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
    #                      columns=[i for i in classes])
    # plt.figure(figsize=(12, 7))
    # cm_fig = sn.heatmap(df_cm, annot=True).get_figure()
    
    # writer.add_figure(tag + '/Training Confusion matrix', cm_fig, epoch)
    
    writer.add_scalars(tag + '/Training vs. Validation F1 Score',
                        {'Training':train_f1,'Validation':val_f1},
                        epoch+1)
    
    writer.add_scalars(tag + '/Training vs. Validation Loss',
                    {'Training':train_ave_loss,'Validation':val_ave_loss},
                    epoch+1)

    writer.add_scalars(tag + '/Training vs. Validation Accuracy',
                {'Training':train_accuracy,'Validation':val_accuracy},
                epoch+1)

    writer.add_scalars(tag + '/Training vs. Validation Recall',
                {'Training':train_recall,'Validation':val_recall},
                epoch+1)
    
    writer.add_scalars(tag + '/Training vs. Validation Precision',
                    {'Training':train_precision,'Validation':val_precision},
                    epoch+1)

    # Double check labels are correct with recall/precision/accuracy calculation
    writer.add_scalars(tag + '/Training TFPN',
                      {'TN':train_cf_matrix[0][0],'FP':train_cf_matrix[0][1],
                       'FN':train_cf_matrix[1][0],'TP':train_cf_matrix[1][1]},
                      epoch+1)

    writer.add_scalars(tag + '/Validation TFPN',
                  {'TN':val_cf_matrix[0][0],'FP':val_cf_matrix[0][1],
                   'FN':val_cf_matrix[1][0],'TP':val_cf_matrix[1][1]},
                  epoch+1)
    writer.flush()

    # checkpoint
    if (epoch+1) % args.chkpnt == 0:
        model_path = '{}{}_checkpoint'.format(MODEL_DIR,tag)
        print("Checkpointing model. Epoch: {}".format(epoch+1))
        torch.save(model.state_dict(), model_path)

    # save best accuracy
    if val_accuracy > val_accuracy_best:
        val_accuracy_best = val_accuracy
        model_path = '{}{}_bestAccuracy'.format(MODEL_DIR,tag)
        print("Saving model. Epoch: {}".format(epoch+1))
        best_accuracy_epoch = epoch+1
        torch.save(model.state_dict(), model_path)

    # save best recall
    if val_recall > val_recall_best:
        val_recall_best = val_recall
        model_path = '{}{}_bestRecall'.format(MODEL_DIR,tag)
        print("Saving model. Epoch: {}".format(epoch+1))
        best_recall_epoch = epoch+1
        torch.save(model.state_dict(), model_path)

print("Best Accuracy Epoch: {}".format(best_accuracy_epoch))
print("Best Recall Epoch: {}".format(best_recall_epoch))
