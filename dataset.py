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


########### Creating train, val, test datapaths ###########

train_val_image_paths = glob.glob(r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\images\train\*.jpg')
train_val_mask_paths = glob.glob(r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\color_labels\train\*.png')

# Creating 6000 train and 1000 val images/masks from the 7000 images in the train directories
len_train_val_image_mask_paths = len(train_val_image_paths)
train_size = 6/7

train_image_paths = train_val_image_paths[:int(len_train_val_image_mask_paths*train_size)]
train_mask_paths = train_val_mask_paths[:int(len_train_val_image_mask_paths*train_size)]

val_image_paths = train_val_image_paths[int(len_train_val_image_mask_paths*train_size):]
val_mask_paths = train_val_mask_paths[int(len_train_val_image_mask_paths*train_size):]

test_image_paths = glob.glob(r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\images\val\*.jpg')
test_mask_paths = glob.glob(r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\color_labels\val\*.png')

########### Checking the image paths ###########
#print(len(train_image_paths))
#print(len(train_mask_paths))
#print(len(val_image_paths))
#print(len(val_mask_paths))
#print(len(test_image_paths))
#print(len(test_mask_paths))

#image = Image.open(train_image_paths[0])
#image.show()
#mask = Image.open(train_mask_paths[0])
#mask.show()

#for i in train_image_paths:
 #   print(i)

#for j in train_mask_paths:
#    print(j)


########### Creating the custom dataset ###########
class Bdd100k_dataset(Dataset):
    def __init__(self, image_paths, target_paths, train=True):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.device = torch.device('cuda:0')

    def transform(self, image, mask):
        
        ### Resize, if ResNet requires it ###
        #resize = torchvision.transforms.Resize(size=(520, 520))
        #imagei = resize(image)
        #maski = resize(mask)

        ### RGBA => RGB Conversion if neccessary ###
        mask_rgb = mask
        if mask.shape == (720, 1280, 4):
            mask_rgb = color.rgba2rgb(mask)
      
        ### Transform Numpy Array to tensor ###
        tensor_image = torch.tensor(image, dtype = torch.float)
        tensor_mask = torch.tensor(mask_rgb, dtype = torch.float)
                
        return tensor_image, tensor_mask

    def __getitem__(self, index):
        image = io.imread(self.image_paths[index])
        mask = io.imread(self.target_paths[index])
        x, y = self.transform(image, mask)
        
        #return {'image':x, 'mask':y}
                
        return x, y

    def __len__(self):
        return len(self.image_paths)




if __name__ == "__main__":

    ########### Creating the DataLoaders ###########
    train_dataset = Bdd100k_dataset(train_image_paths, train_mask_paths)
    #print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)

    val_dataset = Bdd100k_dataset(val_image_paths, val_mask_paths)
    #print(len(val_dataset))
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=1)

    test_dataset = Bdd100k_dataset(test_image_paths, test_mask_paths)
    #print(len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=1)

    ########### Checking the Dimensions ###########
    #for i in range(len(train_dataset)):
    #    sample = train_dataset[i]

    #    print(i, sample['image'].shape, sample['mask'].shape)


    ########### Testing DataLoaders ###########        
    images, masks = next(iter(train_loader))

    #out = torchvision.utils.make_grid(images)  
    
    print(images[0].shape, masks.shape)
    
    DLTNIT = plt.figure('DataLoader Tensor => Numpy Image test')            # Something is wrong with the plotted pictures
    image_numpy = images[0].numpy()    
    print('DataLoader Tensor => Numpy Image test: ', image_numpy.shape)
    plt.imshow(image_numpy)
    DLTNIT.show()

    DLTNMT = plt.figure('DataLoader Tensor => Numpy Mask test')
    mask_numpy = masks[0].numpy()
    print('DataLoader Tensor => Numpy Mask test: ', mask_numpy.shape)
    plt.imshow(mask_numpy)
    DLTNMT.show()
    
    
    ########### Testing RGBA => RGB Conversion ###########
    MTBC = plt.figure('Mask test before RGBA => RGB conversion')
    mask_testing = io.imread(train_mask_paths[0])                   # RGBA mask image!
    print('Mask test before RGBA => RGB conversion: ', mask_testing.shape)
    plt.imshow(mask_testing)
    MTBC.show()

    MTAC = plt.figure('Mask test after RGBA => RGB conversion')
    mask_testing_rgb = mask_testing
    if mask_testing.shape == (720, 1280, 4):                        # Conversion with scikit-image
            mask_testing_rgb = color.rgba2rgb(mask_testing)
    print('Mask test after RGBA => RGB conversion: ', mask_testing_rgb.shape)
    plt.imshow(mask_testing_rgb)
    MTAC.show()
    
    plt.show()