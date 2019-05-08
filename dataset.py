import torch
from torch.utils.data import Dataset, DataLoader
import glob
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import _pickle as pickle
from data_creator import DataCreator                        
from data_creator import read_object
from data_creator import create_mask_image


print('Torch version: ' + str(torch.__version__))
print('GPU available: ' + str(torch.cuda.is_available()))

desktop_train_pickle_read_dir_0_3 = glob.glob(r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\train\\0.3\\*.pkl')
desktop_train_pickle_read_dir_0_4 = glob.glob(r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\train\\0.4\\*.pkl')
desktop_train_pickle_read_dir_0_5 = glob.glob(r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\train\\0.5\\*.pkl')
desktop_train_pickle_read_dir_0_6 = glob.glob(r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\train\\0.6\\*.pkl')
desktop_train_pickle_read_dir_0_8 = glob.glob(r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\train\\0.8\\*.pkl')
desktop_train_pickle_read_dir_1 = glob.glob(r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\train\\1\\*.pkl')
desktop_train_pickle_read_dir_1_5 = glob.glob(r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\train\\1.5\\*.pkl')
desktop_train_pickle_read_dir_2 = glob.glob(r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\train\\2\\*.pkl')
desktop_train_pickle_read_dirs = [desktop_train_pickle_read_dir_0_3,
                            desktop_train_pickle_read_dir_0_4,
                            desktop_train_pickle_read_dir_0_5,
                            desktop_train_pickle_read_dir_0_6]

train_pickle_read_dir_0_3 = glob.glob('/content/drive/My Drive/PSPNet/BDD100K/seg/pickle/train/0.3/*.pkl')
train_pickle_read_dir_0_4 = glob.glob('/content/drive/My Drive/PSPNet/BDD100K/seg/pickle/train/0.4/*.pkl')
train_pickle_read_dir_0_5 = glob.glob('/content/drive/My Drive/PSPNet/BDD100K/seg/pickle/train/0.5/*.pkl')
train_pickle_read_dir_0_6 = glob.glob('/content/drive/My Drive/PSPNet/BDD100K/seg/pickle/train/0.6/*.pkl')
train_pickle_read_dir_0_8 = glob.glob('/content/drive/My Drive/PSPNet/BDD100K/seg/pickle/train/0.8/*.pkl')
train_pickle_read_dir_1 = glob.glob('/content/drive/My Drive/PSPNet/BDD100K/seg/pickle/train/1/*.pkl')
train_pickle_read_dir_1_5 = glob.glob('/content/drive/My Drive/PSPNet/BDD100K/seg/pickle/train/1.5/*.pkl')
train_pickle_read_dir_2 = glob.glob('/content/drive/My Drive/PSPNet/BDD100K/seg/pickle/train/2/*.pkl')
train_pickle_read_dirs = [train_pickle_read_dir_0_3,
                            train_pickle_read_dir_0_4,
                            train_pickle_read_dir_0_5,
                            train_pickle_read_dir_0_6]

desktop_val_pickle_read_dir = glob.glob(r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\val\*.pkl')
val_pickle_read_dir = glob.glob('/content/drive/My Drive/PSPNet/BDD100K/seg/pickle/val/*.pkl')

desktop_test_pickle_read_dir = glob.glob(r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\test\*.pkl')
test_pickle_read_dir = glob.glob('/content/drive/My Drive/PSPNet/BDD100K/seg/pickle/test/*.pkl')

########### Creating the custom dataset ###########
class Bdd100k_dataset(Dataset):
    def __init__(self, mode, pickle_samples, batch_size):
        self.mode = mode
        self.pickle_samples = pickle_samples.copy()        
        self.batch_size = batch_size
        self.device = torch.device('cuda:0')

        if self.mode == 'Train':
            self.sub_sample_number = len(pickle_samples[0]) * len(pickle_samples[0][0])
        elif self.mode == 'Val':
            self.sub_sample_number = len(pickle_samples)
        elif self.mode == 'Test':
            self.sub_sample_number = len(pickle_samples)

    def transform(self, pickle_image, pickle_mask):
                      
        cuda_tensor_image = (pickle_image.type(torch.float32)/255.0).to(device = self.device)             #dtype = torch.cuda.FloatTensor,
        
        one_hot_mask = (np.arange(20) == pickle_mask[...,np.newaxis]).astype(np.uint8)
        cuda_tensor_mask = torch.clamp(torch.abs(torch.tensor(one_hot_mask, dtype = torch.float32).permute(2,0,1)), min = 1e-8, max = 1.0 - 1e-8).to(device = self.device)
        
        #checkpoint = time.time()
        #print(checkpoint - start)
                       
        return cuda_tensor_image, cuda_tensor_mask

    def __getitem__(self, index):

        if self.mode == 'Train':
            obj = self.get_train_item(index = index)
        elif (self.mode == 'Val' or self.mode == 'Test'):
            obj = read_object(self.pickle_samples[index])
        else:
            print('Wrong mode! Please select from the following: Train, Val, Test')
                
        #print(obj.name)
        
        pickle_image = obj.image
        pickle_mask = obj.mask
        
        cuda_tensor_image, cuda_tensor_mask = self.transform(pickle_image, pickle_mask)
        
        return cuda_tensor_image, cuda_tensor_mask

    def get_train_item(self, index):
        current_pickle_samples = self.pickle_samples[index // self.sub_sample_number]        
        current_index = int(index - (index // self.sub_sample_number) * self.sub_sample_number)
        current_sub_index_0 = int(current_index // self.batch_size)
        current_sub_index_1 = int(current_index % self.batch_size)
        
        obj = read_object(current_pickle_samples[current_sub_index_0][current_sub_index_1])

        return obj
                
    def __len__(self): 
        if self.mode == 'Train':
            return len(self.pickle_samples) * len(self.pickle_samples[0]) * len(self.pickle_samples[0][0])

        elif (self.mode == 'Val' or self.mode == 'Test'):
            return self.sub_sample_number
            

def create_train_samples(batch_size, pickle_read_dirs = train_pickle_read_dirs):
    
    shuffeled_0_3_dir = pickle_read_dirs[0].copy()
    shuffeled_0_4_dir = pickle_read_dirs[1].copy()
    shuffeled_0_5_dir = pickle_read_dirs[2].copy()
    shuffeled_0_6_dir = pickle_read_dirs[3].copy()
    
    random.shuffle(shuffeled_0_3_dir)
    random.shuffle(shuffeled_0_4_dir)
    random.shuffle(shuffeled_0_5_dir)
    random.shuffle(shuffeled_0_6_dir)
    
    train_samples_0_3 = [shuffeled_0_3_dir[i:i+batch_size] for i in range(0, int(len(shuffeled_0_3_dir)), batch_size)]
    train_samples_0_4 = [shuffeled_0_4_dir[i:i+batch_size] for i in range(0, int(len(shuffeled_0_4_dir)), batch_size)]
    train_samples_0_5 = [shuffeled_0_5_dir[i:i+batch_size] for i in range(0, int(len(shuffeled_0_5_dir)), batch_size)]
    train_samples_0_6 = [shuffeled_0_6_dir[i:i+batch_size] for i in range(0, int(len(shuffeled_0_6_dir)), batch_size)]
    
    train_samples = [train_samples_0_6, train_samples_0_3, train_samples_0_4, train_samples_0_5]
    
    return train_samples

def create_val_samples(batch_size, pickle_read_dirs = val_pickle_read_dir):
    
    shuffeled_val_dir = pickle_read_dirs.copy()
        
    random.shuffle(shuffeled_val_dir)

    #val_samples = [shuffeled_val_dir[i:i+batch_size] for i in range(0, int(len(shuffeled_val_dir)), batch_size)]
            
    return shuffeled_val_dir

def create_test_samples(batch_size, pickle_read_dirs = test_pickle_read_dir):
    
    shuffeled_test_dir = pickle_read_dirs.copy()
        
    random.shuffle(shuffeled_test_dir)

    #test_samples = [shuffeled_test_dir[i:i+batch_size] for i in range(0, int(len(shuffeled_test_dir)), batch_size)]
            
    return shuffeled_test_dir
    
        
    



if __name__ == "__main__":
    '''
    print(len(desktop_train_pickle_read_dir_0_3))
    print(len(desktop_train_pickle_read_dir_0_4))
    print(len(desktop_train_pickle_read_dir_0_5))
    print(len(desktop_train_pickle_read_dir_0_6))
    print(len(desktop_val_pickle_read_dir))
    print(len(desktop_test_pickle_read_dir))

    batch_size = 4

    train_samples = create_train_samples(batch_size = batch_size, pickle_read_dirs = desktop_train_pickle_read_dirs)
    for i in range(len(train_samples)):
        print(len(train_samples[i]))
        print(len(train_samples[i][i]))

    val_samples = create_val_samples(batch_size = batch_size, pickle_read_dirs = desktop_val_pickle_read_dir)
    print(len(val_samples))

    test_samples = create_test_samples(batch_size = batch_size, pickle_read_dirs = desktop_test_pickle_read_dir)
    print(len(test_samples))

    train_dataset = Bdd100k_dataset(mode = 'Train', pickle_samples = train_samples, batch_size = batch_size)
    print(len(train_dataset))

    val_dataset = Bdd100k_dataset(mode = 'Val', pickle_samples = val_samples, batch_size = batch_size)
    print(len(val_dataset))

    test_dataset = Bdd100k_dataset(mode = 'Test', pickle_samples = test_samples, batch_size = batch_size)
    print(len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
    print(len(train_loader))

    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    print(len(val_loader))

    test_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    print(len(test_loader))    

    
    
    
    it = iter(test_loader)
    
    
    for i in range(250):
        images, masks = next(it)
        print(i)
        print(images.shape)
        print(masks.shape)
    '''
     
    
    '''
    DLTNIT = plt.figure('DataLoader Tensor => Numpy Image test')            
    image_numpy = images[0].cpu().detach().permute(1, 2, 0).numpy()    
    print('DataLoader Tensor => Numpy Image test: ', image_numpy.shape)
    plt.imshow(image_numpy)
    DLTNIT.show()
    

    DLTNMT = plt.figure('DataLoader Tensor => Numpy Mask test')
    mask_complex = masks[0].cpu().detach()
    mask_image = create_mask_image(mask_complex)
    mask_numpy = mask_image.numpy()
    print('DataLoader Tensor => Numpy Mask test: ', mask_numpy.shape)
    plt.imshow(mask_numpy/255.0)
    DLTNMT.show()

    plt.show()
    
    '''
    

   

    
    