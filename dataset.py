from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as F
import glob
import matplotlib.pyplot as plt


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

    def transform(self, image, mask):
        
        ### Resize, if ResNet requires it ###
        # resize = transforms.Resize(size=(520, 520))
        # image = resize(image)
        # mask = resize(mask)
      
        ### Transform PIL Image to tensor ###
        image = F.to_tensor(image)
        mask = F.to_tensor(mask)
        
        return image, mask

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index])
        x, y = self.transform(image, mask)
        return x, y

    def __len__(self):
        return len(self.image_paths)



########### Creating the DataLoaders ###########
if __name__ == "__main__":

    train_dataset = Bdd100k_dataset(train_image_paths, train_mask_paths)
    #print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)

    val_dataset = Bdd100k_dataset(val_image_paths, val_mask_paths)
    #print(len(val_dataset))
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=1)

    test_dataset = Bdd100k_dataset(test_image_paths, test_mask_paths)
    #print(len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=1)

    

 ########### Testing DataLoaders ###########       
    
    images, masks = next(iter(train_loader))  # Some kind of tensor dimension problem..
    
    out = torchvision.utils.make_grid(images)

    ### Testing if the PIL Image -> tensor conversion is the problem.... Seems good to me ###
    image_testing = Image.open(train_image_paths[0])
    image_testing.show()
    mask_testing = Image.open(train_mask_paths[0])
    mask_testing.show()

    image_testing_to_tensor = F.to_tensor(image_testing)
    mask_testing_to_tensor = F.to_tensor(mask_testing)

    image_tensor_to_numpy = image_testing_to_tensor.numpy().transpose((1, 2, 0))
    figure = plt.imshow(image_tensor_to_numpy)
    plt.show()

    mask_tensor_to_numpy = mask_testing_to_tensor.numpy().transpose((1, 2, 0))
    figure = plt.imshow(mask_tensor_to_numpy)
    plt.show()