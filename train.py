from google.colab import drive
drive.mount('/content/drive')


!pip install PyDrive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
# Authenticate and create the PyDrive client.
# This only needs to be done once per notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

file_id_0_3 = '1W6lTjlUpoSwdwxV4fvgkUkFWGECSMysp'
file_id_0_4 = '1TigoM3OZlkP1KKA7dykrveZXz4_te5CE'
file_id_0_5 = '19BCd_0dSYs1lVV5Fr7Cn91KoIwHqyUBe'
file_id_0_6 = '1tSq-y00vpwvZCID38XqTEPzY5SHcgytY'
file_id_val = '1a0wmowv1uI4C_MNIeQlcgwmaR_HV1JWr'

file_ids = [file_id_0_3, file_id_0_4, file_id_0_5, file_id_0_6, file_id_val]
file_names = ['0.3.zip', '0.4.zip', '0.5.zip', '0.6.zip', 'val.zip']

for i in range(len(file_ids)):
  downloaded = drive.CreateFile({'id': file_ids[i]})
  downloaded.GetContentFile(file_names[i])
                          
!unzip 0.3.zip
!unzip 0.4.zip
!unzip 0.5.zip
!unzip 0.6.zip
!unzip val.zip 


import glob
path_0_3 = glob.glob('/content/0.3/*.pkl')
path_0_4 = glob.glob('/content/0.4/*.pkl')
path_0_5 = glob.glob('/content/0.5/*.pkl')
path_0_6 = glob.glob('/content/0.6/*.pkl')
path_val = glob.glob('/content/val/*.pkl')
paths = [path_0_3, path_0_4, path_0_5, path_0_6, path_val]

for i in range(len(paths)):
  print(len(paths[i]))
  print(paths[i][0])


from PIL import Image
from skimage import io
from skimage import color
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as F
import glob
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
import random
import torch.optim as optim
import time
import _pickle as pickle

from data_creator import DataCreator                        
from data_creator import read_object
from data_creator import create_mask_image

from dataset import train_pickle_read_dirs
from dataset import val_pickle_read_dir
from dataset import create_train_samples
from dataset import create_val_samples
from dataset import Bdd100k_dataset

import feature_extractor as extractors

from PSPModule import PSPNet

def train(batch_size = 4, max_iter = 90000):
    since = time.time()

    base_lr_rate = 0.01
    power = 0.9
    #momentum = 0.9
    weight_decay = 0.0001
    aux_factor = 0.4
    starting_epoch = 0
    
    ####################### LOAD CHECKPOINTS #######################
    #CHECKPOINT_PATH = '/content/drive/My Drive/PSPNet/ModelParams/train_valid_pspnet-epoch{}-dsetID{}.pth'.format(5, 67.033)
    #checkpoint = torch.load(CHECKPOINT_PATH)
    #starting_epoch = checkpoint['epoch'] + 1
    
    #DATASETS_PATH = '/content/drive/My Drive/PSPNet/CreatedDatasets/{}.pth'.format(67.033)
    #datasets = torch.load(DATASETS_PATH)
    ################################################################
    
    #train_samples = datasets['Train']
    train_samples = create_train_samples(batch_size = batch_size, pickle_read_dirs = train_pickle_read_dirs)
    train_dataset = Bdd100k_dataset(pickle_samples = train_samples, batch_size = batch_size, mode = 'Train')
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
    
    #val loader és társainak implementálása
    
    #val_samples = datasets['Val']
    val_samples = create_val_samples(batch_size = batch_size, pickle_read_dirs = val_pickle_read_dir)
    val_dataset = Bdd100k_dataset(pickle_samples = val_samples, batch_size = batch_size, mode = 'Val')
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    
    #train_loader_iter = iter(train_loader)
    #val_loader_iter = iter(val_loader)
    
    #dataset_ID = 67.033
    dataset_ID = round(random.uniform(0.0, 100.0), 3)
    DATASETS_PATH = '/content/drive/My Drive/PSPNet/CreatedDatasets/{}.pth'.format(dataset_ID)                        
    torch.save({
                'Train': train_samples,
                'Val': val_samples               
                }, DATASETS_PATH)
    
    val_num_batches = int(len(val_dataset) / batch_size)    
    train_num_batches = int(len(train_dataset) / batch_size)    
    num_epochs = int(max_iter / train_num_batches)
    
    print('-' * 30)
    #print(len(train_samples[0]))
    #print(len(train_dataset))
    print('MAX NUMBER OF ITERATIONS: ' + str(max_iter))
    print('BATCH SIZE: ' + str(batch_size))
    print('NUMBER OF EPOCHS: ' + str(num_epochs))
    print('NUMBER OF TRAINING DATA BATCHES: ' + str(train_num_batches))
    print('NUMBER OF VALIDATION DATA BATCHES: ' + str(val_num_batches))
    print('-' * 30)
   
    
    model = PSPNet()
   # model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()

    criterion = torch.nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr = base_lr_rate, weight_decay = weight_decay)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    #for state in optimizer.state.values():
    #  for k, v in state.items():
    #    if torch.is_tensor(v):
    #        state[k] = v.cuda()

    for current_epoch in range(starting_epoch, num_epochs):
        #print('-' * 30)
        print('EPOCH {}/{}'.format(current_epoch + 1, num_epochs))
        print('-' * 30)
        
        epoch_since = time.time()
        
        current_train_batch = 0
        current_val_batch = 0
        
        current_iter = 0
        train_loss = 0.0
        running_val_loss = 0.0
        average_val_loss = 0.0

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #print('-' * 30)
                print('! TRAINING STEP !')
                print('-' * 30)
                
                model.train()  # Set model to training mode
                
                #for i in range(20):
                for train_data in train_loader:
                  train_batch_since = time.time()
                  
                  current_train_batch += 1
                  current_iter = (current_epoch * train_num_batches) + current_train_batch
                  
                  if((current_train_batch % 1000) == 0 or current_train_batch == 1):
                    #print('-' * 30)
                    print('BATCH ({}) {}/{}'.format((current_epoch + 1), current_train_batch, train_num_batches))
                    print('-' * 30)
                  
                  images, targets = train_data
                  #images, targets = next(train_loader_iter)
                  
                  output, aux_output = model(images)
                  
                  #print('-' * 30)
                  #print('CUDA MEMORY ALLOCATED: ' + str(torch.cuda.memory_allocated(torch.device('cuda:0'))/(1024**3)))
                  #print('-' * 30)            
                  
                  scheduler = poly_lr_scheduler(optimizer = optimizer, init_lr = base_lr_rate, iter = current_iter, lr_decay_iter = 1, 
                                            max_iter = max_iter, power = power)
                  
                  scheduler.zero_grad()
                  
                  #print('-' * 30)
                  #print('ZERO GRAD DONE')
                  #print('-' * 30)
                  
                  main_loss = criterion(output, targets)
                  aux_loss = criterion(aux_output, targets)
                  loss = main_loss + aux_factor * aux_loss  
                  
                  #print('-' * 30)
                  #print('TRAIN LOSS IS CALCULATED: {}'.format(loss))
                  #print('-' * 30)
                                    
                  loss.backward()
                  
                  #print('-' * 30)
                  #print('BACKPROPAGATION DONE')
                  #print('-' * 30)
                  
                  scheduler.step()
                  
                  #print('-' * 30)
                  #print('OPTIMIZATION STEP DONE')
                  #print('-' * 30)
                  
                  #print('-' * 30)
                  #print('CUDA MEMORY ALLOCATED: ' + str(torch.cuda.memory_allocated(torch.device('cuda:0'))/(1024**3)))
                  #print('-' * 30)
                  
                  train_time_elapsed = time.time() - train_batch_since
                  
                  #print('-' * 30)
                  #print('TRAINING BATCH TIME IN SEC: ' + str(train_time_elapsed))
                  #print('-' * 30)
                  
                  #!nvidia-smi
                  
                  train_loss = loss.item()
                  
            elif phase == 'val':
                print('-' * 30)
                print('! VALIDATION STEP !')
                print('-' * 30)
                
                model.eval()
                
                with torch.no_grad():
                  for val_data in val_loader:
                  #for i in range(20):
                    val_batch_since = time.time()
                    
                    current_val_batch += 1
                    
                    #print('-' * 30)
                    #print('BATCH ({}) {}/{}'.format((current_epoch + 1), current_val_batch, val_num_batches))
                    #print('-' * 30)
                    
                    images, targets = val_data
                    #images, targets = next(val_loader_iter)
                  
                    output, aux_output = model(images)
                    
                    loss = criterion(output, targets)
                    
                    #print('-' * 30)
                    #print('VAL LOSS IS CALCULATED ! LOSS = {}'.format(loss))
                    #print('-' * 30)
                    
                    #print('-' * 30)
                    #print('CUDA MEMORY ALLOCATED: ' + str(torch.cuda.memory_allocated(torch.device('cuda:0'))/(1024**3)))
                    #print('-' * 30)
                    
                    val_time_elapsed = time.time() - val_batch_since
                    
                    #print('-' * 30)
                    #print('VALIDATION BATCH TIME IN SEC: ' + str(val_time_elapsed))
                    #print('-' * 30)
                    
                    #!nvidia-smi
                    
                    running_val_loss = running_val_loss + loss.item()
        
        average_val_loss = running_val_loss / len(val_loader)
        
        epoch_time_elapsed = time.time() - epoch_since
        
        PATH = '/content/drive/My Drive/PSPNet/ModelParams/train_valid_pspnet-epoch{}-dsetID{}.pth'.format(current_epoch, dataset_ID)                        
        torch.save({
                    'epoch': current_epoch,
                    'iter': current_iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': average_val_loss,
                    'time_hour': (epoch_time_elapsed / 60 / 60)
                    }, PATH)
        
        print('-' * 30)
        print('TRAINING LOSS: ' + str(train_loss))
        print('AVERAGE VAL LOSS: ' + str(average_val_loss))
        print('EPOCH TIME IN MINUTES: ' + str(epoch_time_elapsed // 60))
        print('EPOCH TIME IN HOURS: ' + str(epoch_time_elapsed // 60 / 60))
        print('-' * 30)     


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter = 1,
                      max_iter = 90000, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    print('ITER = ' + str(iter))
    if iter % lr_decay_iter or iter > max_iter:        
        return optimizer
        

    lr = init_lr*(1 - iter/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer
  
