import torch
import torchvision.transforms as T
import numpy as np 
import matplotlib.pyplot as plt
import _pickle as pickle
import glob
from skimage import io
from PIL import Image
import random



color_dict = {  'road': (128, 64, 128), 
                'sidewalk': (244, 35, 232),
                'building': (70, 70, 70),
                'fence': (190, 153, 153),
                'wall': (102, 102, 156),
                'pole': (153, 153, 153),
                'traffic light': (250, 170, 30),
                'traffic sign': (220, 220, 0),
                'terrain': (152, 251, 152),
                'vegetation': (107, 142, 35),
                'sky': (70, 130, 180),
                'person': (220, 20, 60),
                'rider': (255, 0, 0),
                'bicycle': (119, 11, 32),
                'bus': (0, 60, 100),
                'car': (0, 0, 142),
                'motorcycle': (0, 0, 230),
                'train': (0, 80, 100),
                'truck': (0, 0, 70),
                'other': (0, 0, 0)  }



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

# Creating the pickle file SAVING paths
train_pickle_save_dir_0_3 = r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\train\\0.3\\'
train_pickle_save_dir_0_4 = r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\train\\0.4\\'
train_pickle_save_dir_0_5 = r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\train\\0.5\\'
train_pickle_save_dir_0_6 = r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\train\\0.6\\'
train_pickle_save_dir_0_8 = r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\train\\0.8\\'
train_pickle_save_dir_1 = r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\train\\1\\'
train_pickle_save_dir_1_5 = r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\train\\1.5\\'
train_pickle_save_dir_2 = r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\train\\2\\'
train_pickle_save_dirs = [train_pickle_save_dir_0_3,
                            train_pickle_save_dir_0_4,
                            train_pickle_save_dir_0_5,
                            train_pickle_save_dir_0_6]

val_pickle_save_dir = r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\val\\'
test_pickle_save_dir = r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\test\\'

# Creating the pickle file READING paths
train_pickle_read_dir_0_3 = glob.glob(r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\train\\0.3\\*.pkl')
train_pickle_read_dir_0_4 = glob.glob(r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\train\\0.4\\*.pkl')
train_pickle_read_dir_0_5 = glob.glob(r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\train\\0.5\\*.pkl')
train_pickle_read_dir_0_6 = glob.glob(r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\train\\0.6\\*.pkl')

val_pickle_read_dir = glob.glob(r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\val\*.pkl')
test_pickle_read_dir = glob.glob(r'C:\Users\Tony Stark\Desktop\Szakdolgozat\bdd100k\seg\pickle\test\*.pkl')




class DataCreator:    
    def __init__(self, color_dict, index, image_paths, mask_paths, scale_factor):             
        #self.rand_scale_factor = round(random.uniform(0.5, 2), 2)
        self.scale_factor = scale_factor
        self.rand_mirror = bool(random.getrandbits(1))

        self.name = (image_paths[index].rsplit('.', 1)[0]).rsplit('\\', 1)[1]
        
        self.image = self.getitem(index = index, image_paths = image_paths, mask_paths = mask_paths)['image'] #utf8
        self.mask = self.getitem(index = index, image_paths = image_paths, mask_paths = mask_paths)['mask']
        

    def getitem(self, index, image_paths, mask_paths):
        image = io.imread(image_paths[index])
        raw_mask = io.imread(mask_paths[index])
        tensor_image, indexed_numpy_mask = self.transform(image, raw_mask)
        return {'image': tensor_image, 'mask': indexed_numpy_mask}
    
    def transform(self, image, raw_mask):
        rgb_mask = raw_mask     
        if rgb_mask.shape == (720, 1280, 4):        # Some of the masks have a 4th channel with full 255 values 
            rgb_mask = rgb_mask[:,:,:3]             # Creating the RGB raw mask image by cutting off the 4th channel

        PIL_image = T.ToPILImage()(image)           # Transform the images to PIL in order to be able to make transforms on them
        PIL_mask = T.ToPILImage()(rgb_mask)         #       with torchvision 

        
        resized_image = T.Resize((int(720*self.scale_factor),int(1280*self.scale_factor)))(PIL_image)  # Random resizing the images and masks
        resized_mask = T.Resize((int(720*self.scale_factor),int(1280*self.scale_factor)))(PIL_mask)
         
        
        if  self.rand_mirror == True:
            resized_image = T.functional.hflip(resized_image)               # Random mirroring the images and masks
            resized_mask = T.functional.hflip(resized_mask)
        
        scaled_numpy_image = np.array(resized_image)                # Normalize the image pixel values from [0, 255] to [0.0, 1.0]
        tensor_image = torch.tensor(scaled_numpy_image, dtype = torch.uint8).permute(2, 0, 1)    # Transform the tensor from shape (H, W, C) to (C, H, W)
        
       
        tensor_mask = torch.tensor(np.array(resized_mask)).type(torch.int)  # Mask shape is still (H, W, C) 

        stack = create_stack(dimensions = (tensor_mask.shape[0], tensor_mask.shape[1], 3))                  # Create a (20, H, W, C) shaped color stack 

        encoded_mask = torch.eq(torch.sum(torch.abs(stack - tensor_mask), dim = 3), 0).type(torch.float32)  # Create the one-hot encoded (20, H, W) shaped masks (one-hot encoding in the channel dimension) 

        indexed_mask = torch.argmax(encoded_mask, dim = 0).type(torch.uint8)

        indexed_numpy_mask = indexed_mask.numpy().astype(np.uint8)

        return tensor_image, indexed_numpy_mask

        
# This function is responsible for creating the color stack    
def create_stack(color_dict = color_dict, dimensions = (720, 1280, 3)):
        masks = []
        for objects in color_dict:
            masks.append(torch.tensor(np.full(dimensions, color_dict[objects])))    # Create a 20 element list with mask-shaped tensors (H, W, C), each filled full with one of the color_dict values
        stack = torch.stack(masks, dim = 0)                                         # Create a stack from the list (a tensor with a shape of (20, H, W, C))
        return stack


# This function is responsible for creating mask images (tensors with a shape of (H, W, C)) from one-hot encoded masks (tensors with a shape of (20, H, W))
def create_mask_image(indexed_numpy_mask):
    decoded_numpy_mask = (np.arange(20) == indexed_numpy_mask[...,np.newaxis]).astype(np.uint8)
    complex_mask = torch.tensor(decoded_numpy_mask, dtype = torch.float32).permute(2,0,1)

    unbinded_stack = torch.unbind(create_stack(color_dict = color_dict, dimensions = (complex_mask.shape[1], complex_mask.shape[2], 3)), dim = 0)   # Creating a 20-length tuple from (mask-(H, W)-shaped) color stack 
    unbinded_cplx_mask = torch.unbind(complex_mask, dim = 0)    # Creating a 20-length tuple with tensors of shape (H, W) from the one-hot encoded mask (with a shape of (20, H, W))

    ones_list = []                                                                                                       #
    for i in range(len(unbinded_cplx_mask)):                                                                             # Creating a 20-length list with (mask-(H, W)-shaped) tensors with a shape of (C, H, W) with full 1.0 values
        ones_list.append(torch.tensor(np.ones((3, complex_mask.shape[1], complex_mask.shape[2]))).type(torch.float32))   #  
    
    encoded_list = []                                                           #   
    for j in range(len(unbinded_cplx_mask)):                                    #   Extend the one-hot encoded mask tensors (with a shape of (H, W)) in the channel dimension
        encoded_list.append(ones_list[j]*unbinded_cplx_mask[j])                 #   The result tensors have 3 channels (3, H, W) with the same values for all the 3 channels
                                                                                #   
    for k in range(len(unbinded_cplx_mask)):                                    #   
        encoded_list[k] = encoded_list[k].permute(1, 2, 0).type(torch.int)      #   Transform the result tensors from shape (3, H, W) to shape (H, W, 3)
    
    final_list = []                                                             #
    for l in range(len(unbinded_cplx_mask)):                                    #   If given pixel values of a tensor are ones (in the channel dimension), then this will 
        final_list.append(unbinded_stack[l]*encoded_list[l])                    #       transform the values to the corresponding color_dict pixel values
    
    final_stack = torch.stack(final_list, dim = 0)                              #   Creating a stack (20, H, W, 3) tensor from the 20-length (H, W, 3) shaped tensor list, 
                                                                                #       it can be viewed as 20 (H, W, 3) tensors (cubes) behind each other, 
                                                                                #       which could mean that a pixel has 60 channel values and only 3 of these values are the corresponding color values,
                                                                                #       the rest are all 0s.
    
    mask_image = torch.sum(final_stack, dim = 0)                                #   Sum the tensor pixel values in the channel direction (the 57 pixel values of 0s disappear, and only the color values remain)

    return mask_image

def save_object(obj, file_path):
    filename = file_path + obj.name + '.pkl'       
    with open(filename, 'wb') as output:        #   Overwrites any existing file
        pickle.dump(obj, output, -1)            #   -1 = pickle.HIGHEST_PROTOCOL
        

def read_object(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)        
    return obj
    
        
if __name__ == "__main__":
    '''
    scale_factors = [0.3, 0.4, 0.5, 0.6]
    
    for i in range(len(scale_factors)):
        starting_index = 1500 * i
        for index in range(starting_index, starting_index + 1500):   
            myObj = DataCreator(color_dict = color_dict, index = index, image_paths = train_image_paths, mask_paths = train_mask_paths, scale_factor = scale_factors[i])
            save_object(myObj, train_pickle_save_dirs[i])
            if index % 200 == 0:
                print('Training data saving checkpoint: ' + str(index))
    
    for index in range(len(val_image_paths)):
        myObj = DataCreator(color_dict = color_dict, index = index, image_paths = val_image_paths, mask_paths = val_mask_paths, scale_factor = scale_factors[2])
        save_object(myObj, val_pickle_save_dir)
        if index % 200 == 0:
            print('Validation data saving checkpoint: ' + str(index))
    
    for index in range(len(test_image_paths)):
        myObj = DataCreator(color_dict = color_dict, index = index, image_paths = test_image_paths, mask_paths = test_mask_paths, scale_factor = scale_factors[2])
        save_object(myObj, test_pickle_save_dir)
        if index % 200 == 0:
            print('Test data saving checkpoint: ' + str(index))
    '''
    
    '''
    # For testing purposes
    myObj = read_object(train_pickle_read_dir_0_6[1498])

    print(myObj.name)
    print(train_image_paths[5998])

    print(myObj.image.shape)
    print(myObj.image.type())
    print(myObj.mask.shape)
    print(myObj.mask.dtype)

    
        
    mask_image = create_mask_image(myObj.mask)
    print(mask_image.shape)
    test1 = plt.figure('test_mask')            
    mask_numpy = mask_image.numpy()#/255.0    
    plt.imshow(mask_numpy)
    test1.show()
    
    image = myObj.image.type(torch.float32)
    test2 = plt.figure('test_image')
    image_numpy = image.permute(1,2,0).numpy()/255.0
    plt.imshow(image_numpy)
    
    plt.show()
    '''
    