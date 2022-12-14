# -*- KENAN MORANI - IZMIR DEMOCRACY UNIVERSITY -*-

# Importing Libraries
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import nibabel as nib

import tensorflow as tf
 
from tensorflow import keras
from skimage import morphology
from skimage.feature import canny
#from scipy import ndimage as ndi
from skimage import io
from skimage.exposure import histogram

from PIL import Image as im
import skimage
from skimage.filters import sobel

# Mounting my google drive#
#from google.colab import drive

import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow import keras
from PIL import Image

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Stage 1 : UNET MODEL BUILDING #########
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""# Slicing and Saving"""

## The path here include the image volumes (308MB) 
## and the lung masks (1MB) were extracted from the COVID-19 CT segmentation dataset
dataInputPath = '/home/idu/Desktop/COV19D/segmentation/volumes'
imagePathInput = os.path.join(dataInputPath, 'img/') ## Image volumes were exctracted to this subfolder
maskPathInput = os.path.join(dataInputPath, 'mask/') ## lung masks were exctracted to this subfolder

# Preparing the outputpath for slicing the CT volume from the above data
dataOutputPath = '/home/idu/Desktop/COV19D/segmentation/slices/'
imageSliceOutput = os.path.join(dataOutputPath, 'img/') ## Image volume slices will be placed here
maskSliceOutput = os.path.join(dataOutputPath, 'mask/') ## Annotated masks slices will be placed here

# Slicing only in Z direction
# Slices in Z direction shows the required lung area
SLICE_X = False
SLICE_Y = False
SLICE_Z = True

SLICE_DECIMATE_IDENTIFIER = 3

# Choosing normalization boundaries suitable from the chosen images
HOUNSFIELD_MIN = -1020
HOUNSFIELD_MAX = 2995
HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

# Normalizing the images
def normalizeImageIntensityRange (img):
    img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX
    return (img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE

#nImg = normalizeImageIntensityRange(img)
#np.min(nImg), np.max(nImg), nImg.shape, type(nImg)

# Reading image or mask volume
def readImageVolume(imgPath, normalize=True):
    img = nib.load(imgPath).get_fdata()
    if normalize:
        return normalizeImageIntensityRange(img)
    else:
        return img
    
#readImageVolume(imgPath, normalize=False)
#readImageVolume(maskPath, normalize=False)

# Slicing image and saving
def sliceAndSaveVolumeImage(vol, fname, path):
    (dimx, dimy, dimz) = vol.shape
    print(dimx, dimy, dimz)
    cnt = 0
    if SLICE_X:
        cnt += dimx
        print('Slicing X: ')
        for i in range(dimx):
            saveSlice(vol[i,:,:], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_x', path)
            
    if SLICE_Y:
        cnt += dimy
        print('Slicing Y: ')
        for i in range(dimy):
            saveSlice(vol[:,i,:], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_y', path)
            
    if SLICE_Z:
        cnt += dimz
        print('Slicing Z: ')
        for i in range(dimz):
            saveSlice(vol[:,:,i], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_z', path)
    return cnt

# Saving volume slices to file
def saveSlice (img, fname, path):
    img = np.uint8(img * 255)
    fout = os.path.join(path, f'{fname}.png')
    cv2.imwrite(fout, img)
    print(f'[+] Slice saved: {fout}', end='\r')

# Reading and processing image volumes for TEST images
for index, filename in enumerate(sorted(glob.iglob(imagePathInput+'*.nii.gz'))):
    img = readImageVolume(filename, True)
    print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
    numOfSlices = sliceAndSaveVolumeImage(img, 't'+str(index), imageSliceOutput)
    print(f'\n{filename}, {numOfSlices} slices created \n')

# Reading and processing image mask volumes for TEST masks
for index, filename in enumerate(sorted(glob.iglob(maskPathInput+'*.nii.gz'))):
    img = readImageVolume(filename, False)
    print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
    numOfSlices = sliceAndSaveVolumeImage(img, 't'+str(index), maskSliceOutput)
    print(f'\n{filename}, {numOfSlices} slices created \n')

# Exploring the data 
imgPath = os.path.join(imagePathInput, '1.nii.gz')
img = nib.load(imgPath).get_fdata()
np.min(img), np.max(img), img.shape, type(img)


maskPath = os.path.join(maskPathInput, '1.nii.gz')
mask = nib.load(maskPath).get_fdata()
np.min(mask), np.max(mask), mask.shape, type(mask)

# Showing Mask slice
imgSlice = mask[:,:,20]
plt.imshow(imgSlice, cmap='gray')
plt.show()

# Showing Corresponding Image slice
imgSlice = img[:,:,20]
plt.imshow(imgSlice, cmap='gray')
plt.show()

"""# Training and testing Generator"""

# Define constants
#SEED = 42

### Setting the training and testing dataset for validation of the proposed VGG16-UNET model
### All t0&t1 Z-sliced slices (images and masks) were used for testing

BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
SIZE = IMAGE_HEIGHT = IMAGE_HEIGHT
IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

#### Splitting the data into training and test sets happen manually
### t0 volumes and masks were chosen as test sets
data_dir = '/home/idu/Desktop/COV19D/segmentation/slices/'
data_dir_train = os.path.join(data_dir, 'training')
# The images should be stored under: "data/slices/training/img/img"
data_dir_train_image = os.path.join(data_dir_train, 'img')
# The images should be stored under: "data/slices/training/mask/img"
data_dir_train_mask = os.path.join(data_dir_train, 'mask')

data_dir_test = os.path.join(data_dir, 'test')
# The images should be stored under: "data/slices/test/img/img"
data_dir_test_image = os.path.join(data_dir_test, 'img')
# The images should be stored under: "data/slices/test/mask/img"
data_dir_test_mask = os.path.join(data_dir_test, 'mask')


def create_segmentation_generator_train(img_path, msk_path, BATCH_SIZE):
    data_gen_args = dict(rescale=1./255,
                      featurewise_center=True,
                      featurewise_std_normalization=True,
                      rotation_range=90,
                      width_shift_range=0.2,
                      height_shift_range=0.2,
                      zoom_range=0.3
                        )
    datagen = ImageDataGenerator(**data_gen_args)
    
    img_generator = datagen.flow_from_directory(img_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    msk_generator = datagen.flow_from_directory(msk_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    return zip(img_generator, msk_generator)

# Remember not to perform any image augmentation in the test generator!
def create_segmentation_generator_test(img_path, msk_path, BATCH_SIZE):
    data_gen_args = dict(rescale=1./255)
    datagen = ImageDataGenerator(**data_gen_args)
    
    img_generator = datagen.flow_from_directory(img_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    msk_generator = datagen.flow_from_directory(msk_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    return zip(img_generator, msk_generator)

train_generator = create_segmentation_generator_train(data_dir_train_image, data_dir_train_mask, BATCH_SIZE_TRAIN)
test_generator = create_segmentation_generator_test(data_dir_test_image, data_dir_test_mask, BATCH_SIZE_TEST)


NUM_TRAIN = 745
NUM_TEST = 84

## Choosing number of training epoches
NUM_OF_EPOCHS = 20

def display(display_list):
    plt.figure(figsize=(15,15))
    
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap='gray')
    plt.show()

def show_dataset(datagen, num=1):
    for i in range(0,num):
        image,mask = next(datagen)
        display([image[0], mask[0]])

show_dataset(train_generator, 4)

EPOCH_STEP_TRAIN = 6*NUM_TRAIN // BATCH_SIZE_TRAIN
EPOCH_STEP_TEST = NUM_TEST // BATCH_SIZE_TEST

"""# UNet Model"""

from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

SIZE = 224
IMAGE_HEIGHT = SIZE
IMAGE_WIDTH = SIZE

#### UNET NETWORK

# Building 2D-UNET model
def unet(n_levels, initial_features=32, n_blocks=2, kernel_size=3, pooling_size=2, in_channels=1, out_channels=1):
    inputs = keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, in_channels))
    x = inputs
    
    convpars = dict(kernel_size=kernel_size, activation='relu', padding='same')
    
    #downstream
    skips = {}
    for level in range(n_levels):
        for _ in range(n_blocks):
            x = keras.layers.Conv2D(initial_features * 2 ** level, **convpars)(x)
            x = BatchNormalization()(x)
        if level < n_levels - 1:
            skips[level] = x
            x = keras.layers.MaxPool2D(pooling_size)(x)
            
    # upstream
    for level in reversed(range(n_levels-1)):
        x = keras.layers.Conv2DTranspose(initial_features * 2 ** level, strides=pooling_size, **convpars)(x)
        x = keras.layers.Concatenate()([x, skips[level]])
        for _ in range(n_blocks):
            x = keras.layers.Conv2D(initial_features * 2 ** level, **convpars)(x)
            
    # output
    activation = 'sigmoid' if out_channels == 1 else 'softmax'
    x = BatchNormalization()(x)
    x = keras.layers.Conv2D(out_channels, kernel_size=1, activation=activation, padding='same')(x)
    
    return keras.Model(inputs=[inputs], outputs=[x], name=f'UNET-L{n_levels}-F{initial_features}')

## Making the UNet model with different depth levels
UNet_model = unet(2)  # 2-level depth UNet model
#UNet_model = unet(3)  # 3-level depth UNet model
#UNet_model = unet(4)# # 4-level depth UNet model

UNet_model.summary()

### Hyperparameters tuning
    
from tensorflow.keras.metrics import MeanIoU 

initial_learning_rate = 0.1
def lr_exp_decay(epoch, lr):
    k = 1
    return initial_learning_rate * math.exp(-k*epoch)

import math

from tensorflow.keras.metrics import MeanIoU 

UNet_model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=[#tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
                      tf.keras.metrics.MeanIoU(num_classes = 2),
                      'accuracy'])
UNet_model.fit_generator(generator=train_generator, 
                    steps_per_epoch=EPOCH_STEP_TRAIN, 
                    validation_data=test_generator, 
                    validation_steps=EPOCH_STEP_TEST,
                    epochs=NUM_OF_EPOCHS,
                    callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_exp_decay, verbose=1)]
                    )

#Evaluating the UNet models on the test partition
UNet_model.evaluate(test_generator, steps=EPOCH_STEP_TEST)

#Saving the UNet model models with different depth levels and batch norm()
UNet_model.save('/home/idu/Desktop/COV19D/segmentation/UNet_model.h5')

#Loading saved models
UNet_model = keras.models.load_model('/home/idu/Desktop/COV19D/segmentation/UNet_model-3L-BatchNorm.h5')

# Displaying predicted slices against ground truth
def display(display_list):
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap='gray')
    plt.show()

def show_dataset(datagen, num=1):
    for i in range(0, num):
        image,mask = next(datagen)
        display((image[0], mask[0]))

path = '/home/idu/Desktop/COV19D/segmentation/Segmentation Results/'
def show_prediction(datagen, num=1):
    for i in range(0,num):
        image,mask = next(datagen)
        pred_mask = UNet_model.predict(image)[0] > 0.5
        display([image[0], mask[0], pred_mask])        
        num_classes = 2
        IOU_keras = MeanIoU(num_classes=num_classes)  
        IOU_keras.update_state(mask[0], pred_mask)
        print("Mean IoU =", IOU_keras.result().numpy())

        values = np.array(IOU_keras.get_weights()).reshape(num_classes, num_classes)
        print(values) 
        

show_prediction(test_generator, 12)


results = UNet_model.evaluate(test_generator, steps = EPOCH_STEP_TEST)
print("test loss, test acc:", results)

for name, value in zip(UNet_model.metrics_names, results):
    print(name, ':', value)

        
  ## Segmenting images based on k-means clustering and exctracting lung regions and saving them 
#in the same directory as the original images (in .png format)
  
path = '/home/idu/Desktop/COV19D/segmentation/Segmentation Results/'
def show_prediction(datagen, num=1):
    for i in range(0,num):
        image,mask = next(datagen)
        pred_mask = UNet_model.predict(image)[0] > 0.5
        display([image[0], mask[0], pred_mask])        
        num_classes = 2
        IOU_keras = MeanIoU(num_classes=num_classes)  
        IOU_keras.update_state(mask[0], pred_mask)
        print("Mean IoU =", IOU_keras.result().numpy())

        values = np.array(IOU_keras.get_weights()).reshape(num_classes, num_classes)
        print(values) 
        

show_prediction(test_generator, 12)  
  
## calculating average, min and max dice coeffecient on the test set
GT_path = '/home/idu/Desktop/COV19D/segmentation/slices/test/mask/img'   
pred_path = '/home/idu/Desktop/COV19D/segmentation/slices/test/img/img'

import os

# specify the img directory path
#path = "path/to/img/folder/"

# list files in img directory
files = os.listdir(GT_path)
files2 = os.listdir(pred_path)
print(files)

dicee = []
for file in files:
    # make sure file is an image
      for filee in files2:
        if str(filee) == str(file):
          ## Ground Truth Mask
          p1 = os.path.join(GT_path, file)
          print(p1)
          img = cv2.imread(p1 , 0)      
          img = cv2.resize(img, dim)
          img = img / 255.0
          #img = np.asarray(img).astype(np.bool)

          ## Predicted mask
          p2 = os.path.join(pred_path, filee)
          img2 = cv2.imread(p2, 0)
          img2= cv2.resize(img2, dim)
          img2 = img2 / 255.0
          img2 = img2[None]
          img2 = np.expand_dims(img2, axis=-1)
          img2 = UNet_model.predict(img2) > 0.5
          img2 = np.squeeze(img2)
          #img2 = np.asarray(img2).astype(np.bool)

          value = dice_coef(img, img2)
          print("Dice coeffecient value is", value, "\n") 
          dicee.append(value)
          
dicee = np.array(dicee)

L = len(dicee)
print("Number of Values is", len)

# Taking average of dice values
av=np.mean(dicee)
print ("average dice values is", av)

# Taking maximuim and minimuim of dice values
mx=np.max(dicee)
print ("maximuim dice values is", mx)

mn=np.min(dicee)
print ("minimuim dice values is", mn)




####################^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
####################^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
####################^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ STAGE 2 LUNG EXCTRACTION #^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#UNet_model = tf.keras.models.load_model('/home/idu/Desktop/COV19D/segmentation/UNet_model.h5')

## Comparing the results of predicted masks between public dataset and COV19-CT database
import numpy as np # linear algebra
import pandas as pd # reading and processing of tables
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
#from skimage.util.montage import montage2d
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#import dicom
import scipy.misc


## Ectracting with one example 
file_path1 = '/home/idu/Desktop/COV19D/im/156.jpg'
file_path2 = '/home/idu/Desktop/COV19D/segmentation/slices/img/t2-slice140_z.png'

n1 = cv2.imread(file_path1, 0)
n2 = image2=cv2.imread(file_path2, 0)

image1=cv2.imread(file_path1, 0)
image2=cv2.imread(file_path2, 0)

#cv2.imwrite('/home/idu/Desktop/COV19D/im/156.png', image1)
#file_path11 = '/home/idu/Desktop/COV19D/im/156.png'
#image1=cv2.imread(file_path11, 0)

dim = (224, 224)
n1 = cv2.resize(n1, dim)
n2 = cv2.resize(n2, dim)

image1 = cv2.resize(image1, dim)
image2 = cv2.resize(image2, dim)

n1= n1 * 100.0 / 255.0
image1 = image1 * 100.0 / 255.0

hist,bins = np.histogram(image1.flatten(),256,[0,256])
plt.plot(hist, color = 'b')

hist,bins = np.histogram(image2.flatten(),256,[0,256])
plt.plot(hist, color = 'b')
#image2 = cv2.equalizeHist(image2)  ### Histogram equalization
#image = image.expand_dims(segmented_data, axis=-1)
image1 = image1 / 255.0
image2 = image2 / 255.0
image1 = image1[None]
image2 = image2[None]

pred_mask1 = UNet_model.predict(image1) > 0.5
pred_mask2 = UNet_model.predict(image2) > 0.5
pred_mask1 = np.squeeze(pred_mask1)
pred_mask2 = np.squeeze(pred_mask2)

plt.imshow(pred_mask1)
plt.imshow(pred_mask2)

pred_mask1 = np.asarray(pred_mask1, dtype="uint8")
kernel = np.ones((5, 5), np.uint8)
pred_mask11 = cv2.erode(pred_mask1, kernel, iterations=2)
pred_mask11 = cv2.dilate(pred_mask1, kernel, iterations=2)

plt.imshow(pred_mask1)

# Clear Image border
cleared = clear_border(pred_mask1)
plt.imshow(cleared)

# Label iameg
label_image = label(cleared)
plt.imshow(label_image)

#Keep the labels with 2 largest areas.
    
areas = [r.area for r in regionprops(label_image)]
areas.sort()
if len(areas) > 2:
 for region in regionprops(label_image):
  if region.area < areas[-2]:
   for coordinates in region.coords:                
       label_image[coordinates[0], coordinates[1]] = 0
binary = label_image > 0

plt.imshow(binary)

# Erosion
selem = disk(2)
binary = binary_erosion(binary, selem)
plt.imshow(binary)

# Closure operation with a disk of radius 10. This operation is to keep nodules attached to the lung wall.
selem = disk(10)
binary = binary_closing(binary, selem)
plt.imshow(binary)

#Fill in the small holes inside the binary mask of lungs
edges = roberts(binary)
binary = ndi.binary_fill_holes(edges)
plt.imshow(binary)

## Superimposing the binary image on the original image
#binary = int(binary)
binary=binary.astype(np.uint8)
final = cv2.bitwise_and(n1, n1, mask=binary)
plt.imshow(final)

dim = (224, 224)

#kernel = np.ones((5, 5), np.uint8)

### Exctracting for all CT image in COV19-CT-DB
folder_path = '/home/idu/Desktop/COV19D/validation/non-covid' # Changoe this directory to loop over all training, validation and testing images
directory = '/home/idu/Desktop/COV19D/val-seg5/non-covid'  # Changoe this directory to save the lung segmented images in the appropriate bath syncronizing with line above
for fldr in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, fldr)
        dir = os.path.join(directory, fldr)
        os.mkdir(dir)
        for filee in os.listdir(sub_folder_path):
            file_path = os.path.join(sub_folder_path, filee)
            n = cv2.imread(file_path, 0)
            image = cv2.imread(file_path, 0)
            #cv2.imwrite('/home/idu/Desktop/COV19D/im/156.png', image1)
            #file_path11 = '/home/idu/Desktop/COV19D/im/156.png'
            #image1=cv2.imread(file_path11, 0)
            n = cv2.resize(n, dim)
            image = cv2.resize(image, dim)
            image = image * 100.0 / 255.0  ## Squeezing the histogram bins to values intensity between 0 and 100 to make it similar to the histograms in the public dataset
            image = image / 255.0
            image = image[None]
            print('predicting')
            pred_mask = UNet_model.predict(image) > 0.5
            pred_mask = np.squeeze(pred_mask)
            #plt.imshow(pred_mask1)
            pred_mask = np.asarray(pred_mask, dtype="uint8")
             #plt.imshow(pred_mask1)
            # Clear Image border
            cleared = clear_border(pred_mask)
            #plt.imshow(cleared)
            # Label iameg
            label_image = label(cleared)
            #plt.imshow(label_image)
            #Keep the labels with 2 largest areas.    
            areas = [r.area for r in regionprops(label_image)]
            areas.sort()
            if len(areas) > 2:
             for region in regionprops(label_image):
              if region.area < areas[-2]:
               for coordinates in region.coords:                
                 label_image[coordinates[0], coordinates[1]] = 0
            binary = label_image > 0
            #plt.imshow(binary)
            # Erosion
            selem = disk(2)
            binary = binary_erosion(binary, selem)
            #plt.imshow(binary)
            # Closure operation with a disk of radius 10. This operation is to keep nodules attached to the lung wall.
            selem = disk(10)
            binary = binary_closing(binary, selem)
            #plt.imshow(binary)
            #Fill in the small holes inside the binary mask of lungs
            edges = roberts(binary)
            binary = ndi.binary_fill_holes(edges)
            #plt.imshow(binary)
            ## Superimposing the binary image on the original image
            binary=binary.astype(np.uint8)
            final = cv2.bitwise_and(n, n, mask=binary)
            #plt.imshow(final)    
            file_name, file_ext = os.path.splitext(filee) 
            print(fldr)
            dirr = os.path.join(directory, fldr)   
            name=dirr+'/'+str(file_name)+'.jpg'
            print(name)
            print(file_path)
            #final = im.fromarray(final)
            #directory = sub_folder_path
            #name=directory+'/'+str(file_name)+'.png'
            #print(os.path.join(directory, file_name))
            #final.save(name)
            #print(os.path.join(directory, file_name))
            #pth=
            #dir = os.path.join(directory, fldr)
            
            cv2. imwrite(name, final)
            #final.save('{}.png'.format(file_name))
            #print (directory,'',file_name)
            n = []
            image = []



####################^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
####################^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
####################^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ STAGE 3 CLASSIFICATION #^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

####################^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
####################^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Slice Removal (OPtional) #^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
import skimage
from skimage import color, filters
import numpy as np
import os, glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import layers, models
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential
from termcolor import colored  

#import visualkeras
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
#from collections import defaultdict

#from PIL import ImageFont

from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

import pandas as pd

import csv
from sklearn.utils import class_weight 
from collections import Counter

#from tensorflow.keras.models import model_from_json
from keras.callbacks import ModelCheckpoint


file_path1 = '/home/idu/Desktop/COV19D/ct_scan165/0.jpg'
file_path4 = '/home/idu/Desktop/COV19D/ct_scan165/40.jpg'
file_path2 = '/home/idu/Desktop/COV19D/ct_scan165/110.jpg'
file_path3 = '/home/idu/Desktop/COV19D/ct_scan165/235.jpg'
file_path5 = '/home/idu/Desktop/COV19D/ct_scan165/185.jpg'

n1 = cv2.imread(file_path1, 0)
n4 = cv2.imread(file_path4, 0)
n2 = cv2.imread(file_path2, 0)
n3 = cv2.imread(file_path3, 0)
n5 = cv2.imread(file_path5, 0)

dim = (224, 224)
#n1 = cv2.resize(n1, dim)
#n2 = cv2.resize(n2, dim)
#n3 = cv2.resize(n3, dim)
#n4 = cv2.resize(n4, dim)
#n5 = cv2.resize(n5, dim)

#n1 = cv.equalizeHist(n1)
#n2 = cv.equalizeHist(n2)
#n3 = cv.equalizeHist(n3)

#n1 = n1 * 10000.0
#n2 = n2 * 10000.0
#n3 = n3 * 10000.0

hist,bins = np.histogram(n1.flatten(),256,[0,256])
plt.plot(hist, color = 'b')

hist,bins = np.histogram(n2.flatten(),256,[0,256])
plt.plot(hist, color = 'b')

hist,bins = np.histogram(n3.flatten(),256,[0,256])
plt.plot(hist, color = 'b')
#image2 = cv2.equalizeHist(image2)  ### Histogram equalization

# None-representative 
## [Uppermost]
count1 = np.count_nonzero(n1)
print(count1)

count4 = np.count_nonzero(n4)
print(count4)

## [Lowermost]
count3 = np.count_nonzero(n3)
print(count3)

count5 = np.count_nonzero(n5)
print(count5)

# Representative
count2 = np.count_nonzero(n2)
print(count2)

### Chosen threshod is 3500 out of 224*224 = 50176

count = []
folder_path = '/home/idu/Desktop/COV19D/train-seg4/covid' 
#Change this directory to the directory where you need to do preprocessing for images
#Inside the directory must folder(s), which have the images inside them
for fldr in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, fldr)
        files_left = len(directory) # get initial count
        for filee in os.listdir(sub_folder_path):
            file_path = os.path.join(sub_folder_path, filee)
            img = cv2.imread(file_path, 0)
            count = np.count_nonzero(img)  ### COunting number of bright pixels in the binarized slices
            #print(count)
            if count > 2500:  ## The threshold 1500 or 2500
             img = np.expand_dims(img, axis=2)
             img = array_to_img (img)
               # Replace images with the image that includes ROI
             img.save(file_path, 'JPEG')
             #print('saved')
            else:
             if files_left > 1: # check if you should remove
                os.remove(file_path)
                files_left -= 1
            if not os.listdir(sub_folder_path):
              print(sub_folder_path, "Directory is empty")
            count = []


### Using imagedatagenerator to generate images
h= 224
w=224
batch_size = 128
height = h
width = w = 224
train_datagen = ImageDataGenerator(rescale=1./255, 
                              vertical_flip=True,
                              horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/train-seg5/',  ## COV19-CT-DB Training set
        target_size=(h, w),
        batch_size=batch_size,
        color_mode='grayscale',
        classes = ['covid','non-covid'],
        class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/val-seg5/',  ## COV19-CT-DB Validation set
        target_size=(h, w),
        batch_size=batch_size,
        color_mode='grayscale',
        classes = ['covid','non-covid'],
        class_mode='binary')

#### The CNN model

def make_model():
   
    model = tf.keras.models.Sequential()
    
    # Convulotional Layer 1
    model.add(layers.Conv2D(16,(3,3),input_shape=(h,w,1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 2
    model.add(layers.Conv2D(32,(3,3), padding="same"))  
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 3
    model.add(layers.Conv2D(64,(3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())   
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 4
    model.add(layers.Conv2D(128,(3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.1))
    
    # Dense Layer  
    model.add(layers.Dense(1, activation='sigmoid'))
    
    
    return model

model = make_model()

## Choosing number of epoches
n_epochs= 60

# Compiling the model using SGD optimizer with a learning rate schedualer
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
              metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),'accuracy'])

early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=7)

###Learning Rate decay
def decayed_learning_rate(step):
  return initial_learning_rate * decay_rate ^ (step / decay_steps)

initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

## Class weight (Optional)
counter = Counter(train_generator.classes)                          
max_val = float(max(counter.values()))       
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}  
class_weights

## Modify the number if slice removal is used
training_steps = 434726 // batch_size 
val_steps = 88555 // batch_size

history=model.fit(train_generator,
                  steps_per_epoch=training_steps,
                  validation_data=val_generator,
                  validation_steps=val_steps,
                  verbose=2,
                  epochs=n_epochs,
                  callbacks=[early_stopping_cb], #, checkpoint],
                  #class_weight=class_weights
                  )
                  
# Saving the trained CNN model
model.save('/home/idu/Desktop/COV19D/model.h5')

# LOading the CNN model
model = keras.models.load_model('/home/idu/Desktop/COV19D/model.h5')


##Evaluating the CNN model
print (history.history.keys())
            
Train_accuracy = history.history['accuracy']
print(Train_accuracy)
print(np.mean(Train_accuracy))
val_accuracy = history.history['val_accuracy']
print(val_accuracy)
print( np.mean(val_accuracy))

epochs = range(1, len(Train_accuracy)+1)
plt.figure(figsize=(12,6))
plt.plot(epochs, Train_accuracy, 'g', label='Training acc')
plt.plot(epochs, val_accuracy, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.ylim(0.7,1)
plt.xlim(0,50)
plt.legend()

plt.show()

val_recall = history.history['val_recall_1']
print(val_recall)
avg_recall = np.mean(val_recall)
avg_recall

val_precision = history.history['val_precision_1']
avg_precision = np.mean(val_precision)
avg_precision

epochs = range(1, len(Train_accuracy)+1)
plt.figure(figsize=(12,6))
plt.plot(epochs, val_recall, 'g', label='Validation Recall')
plt.plot(epochs, val_precision, 'b', label='Validation Prcision')
plt.title('Validation recall and Validation Percision')
plt.xlabel('Epochs')
plt.ylabel('Recall and Precision')
plt.legend()
plt.ylim(0.5,1)

plt.show()

Macro_F1score = (2*avg_precision*avg_recall)/ (avg_precision + avg_recall)
Macro_F1score


## Making diagnosis predicitons at patient level using the trained CNN model classifier
## Use this for validation set and test set of COV19-DT-DB or other datasets you wish to test the model on
from termcolor import colored

## Choosing the directory where the test/validation data is at
folder_path = '/home/idu/Desktop/COV19D/val-seg3/non-covid'
extensions0 = []
extensions1 = []
extensions2 = []
extensions3 = []
extensions4 = []
extensions5 = []
extensions6 = []
extensions7 = []
extensions8 = []
extensions9 = []
extensions10 = []
extensions11 = []
extensions12 = []
extensions13 = []
covidd = []
noncovidd = []
coviddd = []
noncoviddd = []
covidddd = []
noncovidddd = []
coviddddd = []
noncoviddddd = []
covidd6 = []
noncovidd6 = []
covidd7 = []
noncovidd7 = []
covidd8 = []
noncovidd8 = []
results =1
for fldr in os.listdir(folder_path):
   if fldr.startswith("ct"):
    sub_folder_path = os.path.join(folder_path, fldr)
    for filee in os.listdir(sub_folder_path):
        file_path = os.path.join(sub_folder_path, filee)
        c = cv2.imread(file_path, 0) 
        c = c / 255.0
        #c=img_to_array(c)
        c = np.expand_dims(c, axis=-1)
        c = c[None]
        
        result = model.predict_proba(c) #Probability of 1 (non-covid)
        if result > 0.97:  # Class probability threshod is 0.97
           extensions1.append(results)
        else:
           extensions0.append(results)
        if result > 0.90:  # Class probability threshod is 0.90 
           extensions3.append(results)
        else:
           extensions2.append(results) 
        if result > 0.70:   # Class probability threshod is 0.70
           extensions5.append(results)
        else:
           extensions4.append(results)
        if result > 0.40:   # Class probability threshod is 0.40
           extensions7.append(results)
        else:
           extensions6.append(results)
        if result > 0.50:   # Class probability threshod is 0.50
           extensions9.append(results)
        else:
           extensions8.append(results)
        if result > 0.15:   # Class probability threshod is 0.15
           extensions11.append(results)
        else:
           extensions10.append(results)  
        if result > 0.05:   # Class probability threshod is 0.05
           extensions13.append(results)
        else:
           extensions12.append(results)
    #print(sub_folder_path, end="\r \n")
    ## The majority voting at Patient's level
    if len(extensions1) >  len(extensions0):
      print(fldr, colored("NON-COVID", 'red'), len(extensions1), "to", len(extensions0))
      noncovidd.append(fldr)  
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions0), "to", len(extensions1))
      covidd.append(fldr)    
    if len(extensions3) >  len(extensions2):
      print (fldr, colored("NON-COVID", 'red'), len(extensions3), "to", len(extensions2))
      noncoviddd.append(fldr)
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions2), "to", len(extensions3))
      coviddd.append(fldr)
    if len(extensions5) >  len(extensions4):
      print (fldr, colored("NON-COVID", 'red'), len(extensions5), "to", len(extensions4))
      noncovidddd.append(fldr)
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions5), "to", len(extensions4))
      covidddd.append(fldr)
    if len(extensions7) >  len(extensions6):
      print (fldr, colored("NON-COVID", 'red'), len(extensions7), "to", len(extensions6))
      noncoviddddd.append(fldr)
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions6), "to", len(extensions7))
      coviddddd.append(fldr)
    if len(extensions9) >  len(extensions8):
      print (fldr, colored("NON-COVID", 'red'), len(extensions9), "to", len(extensions8))
      noncovidd6.append(fldr)
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions8), "to", len(extensions9))
      covidd6.append(fldr)
    if len(extensions11) >  len(extensions10):
      print (fldr, colored("NON-COVID", 'red'), len(extensions11), "to", len(extensions10))
      noncovidd7.append(fldr)
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions10), "to", len(extensions11))
      covidd7.append(fldr)
    if len(extensions13) > len(extensions12):
      print (fldr, colored("NON-COVID", 'red'), len(extensions13), "to", len(extensions12))
      noncovidd8.append(fldr)
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions12), "to", len(extensions13))
      covidd8.append(fldr)
       
    extensions0=[]
    extensions1=[]
    extensions2=[]
    extensions3=[]
    extensions4=[]
    extensions5=[]
    extensions6=[]
    extensions7=[]
    extensions8=[]
    extensions9=[]
    extensions10=[]
    extensions11=[]
    extensions12=[]
    extensions13=[]

#Checking the results
print(len(covidd))
print(len(coviddd))
print(len(covidddd))
print(len(coviddddd))
print(len(covidd6))
print(len(covidd7))
print(len(covidd8))
print(len(noncovidd))
print(len(noncoviddd))
print(len(noncovidddd))
print(len(noncoviddddd))
print(len(noncovidd6))
print(len(noncovidd7))
print(len(noncovidd8))
print(len(covidd+noncovidd))
print(len(coviddd+noncoviddd))
print(len(covidddd+noncovidddd))
print(len(coviddddd+noncoviddddd))
print(len(covidd6+noncovidd6))
print(len(covidd7+noncovidd7))
print(len(covidd8+noncovidd8))


## Saving to csv files format to report the results
## Using Majority Voting for each CT scan
import csv

####0.5 slice level class probability 
with open('/home/idu/Desktop/noncovid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(noncovidd6)

with open('/home/idu/Desktop/covid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(covidd6)

####0.9 Slice level class probability
with open('/home/idu/Desktop/noncovid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(noncoviddd)

with open('/home/idu/Desktop/ncovid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(coviddd)

####0.15 Slice level class probability
with open('/home/idu/Desktop/noncovid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(noncovidd7)

with open('/home/idu/Desktop/covid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(covidd7)
############## 0.4 Slice level class probability
with open('/home/idu/Desktop/noncovid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(noncoviddddd)

with open('/home/idu/Desktop/covid.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(coviddddd)
 
 ### KENAN MORANI - THE END OF THE CODE