# -*- KENAN MORANI - IZMIR DEMOCRACY UNIVERSITY -*-
#### COV19-CT DB Database #####
### part of IEEE ICASSP 2023: AI-enabled Medical Image Analysis Workshop and Covid-19 Diagnosis Competition (AI-MIA-COV19D)
### at https://mlearn.lincoln.ac.uk/icassp-2023-ai-mia/
#### B. 3rd COV19D Competition ---- I. Covid-19 Detection Challenge
#### kenan.morani@gmail.com 

# Importing Libraries
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import nibabel as nib

import tensorflow as tf

from tensorflow import keras

from skimage.feature import canny
#from scipy import ndimage as ndi
from skimage import io
from skimage.exposure import histogram

from PIL import Image as im
import skimage
from skimage import data,morphology
from skimage.color import rgb2gray
#import scipy.ndimage as nd

from tensorflow.keras.preprocessing.image import ImageDataGenerator 

from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
#from tensorflow.keras.applications import VGG16
from keras.callbacks import ModelCheckpoint

from skimage import color, filters

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import layers, models
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential
from termcolor import colored  

#import visualkeras
from collections import defaultdict

from PIL import ImageFont

from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

import cv2
import csv
from sklearn.utils import class_weight 
from collections import Counter

from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
#from skimage.util.montage import montage2d
from scipy import ndimage as ndi

#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#import dicom
import scipy.misc

#from tensorflow.keras.models import model_from_json

#import visualkeras
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
#from collections import defaultdict

#from PIL import ImageFont


####################################################################################################
####################### Processing : Slicing and Saving Exctracted Slices from 3D Images [If needed] #########
###############################################################################



"""# Slicing and Saving"""

## The path here include the image volumes (308MB) 
## and the lung masks (1MB) were extracted from the COVID-19 CT segmentation dataset
dataInputPath = '/home/idu/Desktop/COV19D/segmentation/volumes'
imagePathInput = os.path.join(dataInputPath, 'img/') ## Image volumes were exctracted to this subfolder
maskPathInput = os.path.join(dataInputPath, 'mask/') ## lung masks were exctracted to this subfolder

# Preparing the outputpath for slicing the CT volume from the above data++++
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

#BATCH_SIZE_TRAIN = 32
#BATCH_SIZE_TEST = 32

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

BATCH_SIZE = 64

def orthogonal_rot(image):
    return np.rot90(image, np.random.choice([-1, 0, 1]))

    

def create_segmentation_generator_train(img_path, msk_path, BATCH_SIZE):
    data_gen_args = dict(rescale=1./255,
                      featurewise_center=True,
                      featurewise_std_normalization=True,
                      rotation_range=90,
                      preprocessing_function=orthogonal_rot,
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

BATCH_SIZE_TRAIN = BATCH_SIZE_TEST = BATCH_SIZE
SEED = 44
train_generator = create_segmentation_generator_train(data_dir_train_image, data_dir_train_mask, BATCH_SIZE_TRAIN)
test_generator = create_segmentation_generator_test(data_dir_test_image, data_dir_test_mask, BATCH_SIZE_TEST)


NUM_TRAIN = 2*745
NUM_TEST = 2*84

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

show_dataset(test_generator, 2)




#############################################################
########################  Method 1
##################################################################################
##################################################################################
#############################################################
########################  Image Processing and CNN modelling
##################################################################################




### Manual Cropping of the image; r
img = cv2.imread('/home/idu/Desktop/COV19D/val-seg1 (copy)/non-covid/ct_scan210/1.jpg')
img = skimage.color.rgb2gray(img)
r = cv2.selectROI(img)

# Plotting hostogram
# find frequency of pixels in range 0-255
img = Image.open('/home/idu/Desktop/COV19D/val-seg1 (copy)/non-covid/ct_scan210/166.jpg')
img = skimage.color.rgb2gray(img) 
img = img < t
img = np.array(img)
#img = skimage.filters.gaussian(img, sigma=1.0)
# Flatten the image array to 1D
img = img.ravel()

# Plot the histogram
plt.hist(img)#, bins=256, range=(0, 256), color='red', alpha=0.4)
plt.xlabel("Pixel value")
#plt.xlim(0, 50)
plt.ylabel("Counts")
plt.title("Image histogram")
plt.show()

histr = cv2.calcHist([img],[0],None,[256],[0,256])  
# show the plotting graph of an image
plt.plot(histr)
plt.show()

t = 0.45 #Histogram Threshold
#### Cropping right-lung as an ROI and removing upper and lowermost of the slices 
count = []
folder_path = '/home/idu/Desktop/COV19D/val-seg1 (copy)/covid1' 
#Change this directory to the directory where you need to do preprocessing for images
#Inside the directory must folder(s), which have the images inside them
for fldr in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, fldr)
        for filee in os.listdir(sub_folder_path):
            file_path = os.path.join(sub_folder_path, filee)
            img = cv2.imread(file_path)
            #Grayscale images
            img = skimage.color.rgb2gray(img) 
            # First cropping an image
            #%r = cv2.selectROI(im) 
            #Select ROI from images before you start the code 
            #Reference: https://learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/
            #{Last access 15th of Dec, 2021}
            # Crop image using r
            #img_cropped = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            img_cropped=img
            # Thresholding and binarizing images
            # Reference: https://datacarpentry.org/image-processing/07-thresholding/
            #{Last access 15th of Dec, 2021}
            # Gussian Filtering
            #img = skimage.filters.gaussian(img_cropped, sigma=1.0)
            # Binarizing the image
            #print (img)
            img = img < t
            count = np.count_nonzero(img)
            #print(count)
            if count > 260000: ## Threshold to be selected
             #print(count)
             img_cropped = np.expand_dims(img_cropped, axis=2)
             img_cropped = array_to_img (img_cropped)
               # Replace images with the image that includes ROI
             img_cropped.save(str(file_path), 'JPEG')
             #print('saved')
            else:
             #print(count)
                # Remove non-representative slices
             os.remove(str(file_path))
             print('removed')
             print(str(sub_folder_path))
             # Check that there is at least one slice left
            if not os.listdir(str(sub_folder_path)):
              print(str(sub_folder_path), "Directory is empty")
            count = []

#################### Building CNN Model

h = w = 224

#batch_size = 64
batch_size=128
#batch_size=256

train_datagen = ImageDataGenerator(rescale=1./255, 
                              vertical_flip=True,
                              horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/train-Preprocessed/',  ## COV19-CT-DB Training set
        target_size=(h, w),
        batch_size=batch_size,
        color_mode='grayscale',
        classes = ['covid','non-covid'],
        class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/validation-Preprocessed/',  ## COV19-CT-DB Validation set
        target_size=(h, w),
        batch_size=batch_size,
        color_mode='grayscale',
        classes = ['covid','non-covid'],
        class_mode='binary')

#### The CNN model

def make_model():
   
    model = models.Sequential()
    
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
    model.add(layers.Dropout(0.1)) # Try 0.1 or 0.3
    
    # Dense Layer  
    model.add(layers.Dense(1,activation='sigmoid'))
    
    
    return model

model = make_model()


## Load model weights after saving
model.load_weights('/home/idu/Desktop/COV19D/ChatGPT-saved-models/imagepreprocesscnnclass.h5')

n_epochs= 100

# Compiling the model using SGD optimizer with a learning rate schedualer
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
                                  'accuracy'])

early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=7)

checkpoint = ModelCheckpoint('/home/idu/Desktop/COV19D/ChatGPT-saved-models/imagepreprocesscnnclass.h5', save_best_only=True, save_weights_only=True)

###Learning Rate decay

initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

def decayed_learning_rate(step):
  return initial_learning_rate * decay_rate ^ (step / decay_steps)

## Class weight
counter = Counter(train_generator.classes)                          
max_val = float(max(counter.values()))       
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}  
class_weights

training_steps = 2*269309 // batch_size
training_steps = 434726 // batch_size ## Without Slice reduction
val_steps = 67285 // batch_size
val_steps = 106378 // batch_size ## Without Slice reduction

history=model.fit(train_generator,
                  steps_per_epoch=training_steps,
                  validation_data=val_generator,
                  validation_steps=val_steps,
                  verbose=2,
                  epochs=n_epochs,
                  callbacks=[early_stopping_cb, checkpoint],
                  class_weight=class_weights)

## saving the model
model.save('/home/idu/Desktop/COV19D/ChatGPT-saved-models/imagepreprocesscnnclass.h5')

model = keras.models.load_model('/home/idu/Desktop/COV19D/ChatGPT-saved-models/imagepreprocesscnnclass.h5')

# Evaluatin the model
model.evaluate(val_generator, batch_size=128)



  

#############################################################
########################  Method 2 & 3
##################################################################################
##################################################################################
#############################################################
########################  Stage 1 : SEGMENTAITON
##################################################################################


    
############ K-Means Clustering Based Segmetnation [Optional]

def extract_lungs(mask):
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def kmeans_segmentation(image):
    Z = image.reshape((-1,1))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    segmented_image = res.reshape((image.shape))
    return segmented_image

def segment_and_extract_lungs(image):
    segmented_image = kmeans_segmentation(image)
    binary_mask = np.zeros(image.shape, dtype=np.uint8)
    binary_mask[segmented_image == segmented_image.min()] = 1
    binary_mask = extract_lungs(binary_mask)
    lung_extracted_image = np.zeros(image.shape, dtype=np.uint8)
    lung_extracted_image[binary_mask == 1] = image[binary_mask == 1]
    return lung_extracted_image

# Modify here to run the code on all the required slices
input_folder = "/home/idu/Desktop/COV19D/train/non-covid"
output_folder = "/home/idu/Desktop/COV19D/train-seg1/non-covid"


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for subdir, dirs, files in os.walk(input_folder):
    for file in files:
        image_path = os.path.join(subdir, file)
        if '.jpg' in image_path:
            image = cv2.imread(image_path, 0)
            lung_extracted_image = segment_and_extract_lungs(image)
            subfolder_name = subdir.split('/')[-1]
            subfolder_path = os.path.join(output_folder, subfolder_name)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
            output_path = os.path.join(subfolder_path, file)
            cv2.imwrite(output_path, lung_extracted_image)


################# Remove Non-representative Slices [optional]

def check_valid_image(image):
    return cv2.countNonZero(image) > 1764 # Choose a threshold for removal

input_folder = "/home/idu/Desktop/COV19D/train-seg/non-covid"
output_folder = "/home/idu/Desktop/COV19D/train-seg-sliceremove/non-covid"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for subdir, dirs, files in os.walk(input_folder):
    subfolder_name = subdir.split('/')[-1]
    subfolder_path = os.path.join(output_folder, subfolder_name)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    count = 0
    for file in files:
        image_path = os.path.join(subdir, file)
        if '.jpg' in image_path:
            image = cv2.imread(image_path, 0)
            if check_valid_image(image):
                count += 1
                output_path = os.path.join(subfolder_path, file)
                cv2.imwrite(output_path, image)
    if count == 0:
        print(f"No valid images were found in subfolder: {subfolder_name}")


# calculating average, min and max dice coeffecient on the test set
GT_path = '/home/idu/Desktop/COV19D/segmentation/slices/test/mask/img'   
pred_path = '/home/idu/Desktop/COV19D/segmentation/slices/test/img/img'

### Mean IoU & Dice COeffecient Measures

# specify the img directory path
#path = "path/to/img/folder/"

# list files in img directory
files = os.listdir(GT_path)
files2 = os.listdir(pred_path)
print(files)

mean_iou = []
dicee = []
dim = (224,224)
num_classes = 2
for file in files:
    # make sure file is an image
      for filee in files2:
        if str(filee) == str(file):
          ## Ground Truth Mask
          p1 = os.path.join(GT_path, file)
          print(p1)
          img = cv2.imread(p1 , 0)      
          img = cv2.resize(img, dim)
          edges = canny(img)
          img = nd.binary_fill_holes(edges)
          elevation_map = sobel(img)
          # Since, the contrast difference is not much. Anyways we will perform it
          markers = np.zeros_like(img)
          markers[img < 0.1171875] = 1 # 30/255
          markers[img > 0.5859375] = 2 # 150/255
          segmentation = morphology.watershed(elevation_map, markers)
          #img = img / 255.0
          #imgg = img
          
                   #img = numpy.bool(img)
          #img = np.asarray(img).astype(np.bool)

          ## Predicted mask
          #p2 = os.path.join(pred_path, filee)
          #img2 = cv2.imread(p2, 0)
          #img2= cv2.resize(img2, dim)
          #img2 = img2 / 255.0
          #img2 = img2[None]
          #img2 = np.expand_dims(img2, axis=-1)
          #img2 = UNet_model.predict(img2) > 0.5
          #imgg2 = UNet_model.predict(img2) #> 0.5
          #imgg2 = imgg2.astype(numpy.float64)
          #img2 = np.squeeze(img2)
          #imgg2 = np.squeeze(imgg2)
          #d = dtype(image)
          #print(d)
          #img2 = np.asarray(img2).astype(np.bool)
          
          IOU_keras = MeanIoU(num_classes=num_classes)
          IOU_keras.update_state(img, segmentation)
          print("Mean IoU =", IOU_keras.result().numpy())
          mean_iou.append(IOU_keras.result().numpy())

          value = dice_coef(img, segmentation)
          print("Dice coeffecient value is", value, "\n") 
          dicee.append(value)
          
 
          

############ UNET Based Segemtnation 

# Building 2D-UNET model
def unet(n_levels, initial_features=64, n_blocks=2, kernel_size=5, pooling_size=2, in_channels=1, out_channels=1):
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

## Choosing UNet depth 
#UNet_model = unet(2)  # 2-level depth UNet model
UNet_model = unet(3)  # 3-level depth UNet model
#UNet_model = unet(4)# # 4-level depth UNet model

UNet_model.summary()

# Hyperparameters tuning
   
from tensorflow.keras.metrics import MeanIoU 
import math

initial_learning_rate = 0.1
def lr_exp_decay(epoch, lr):
    k = 1
    return initial_learning_rate * math.exp(-k*epoch)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    #verbose=1,
    #mode="auto",
    #baseline=None,
    #restore_best_weights=False,
)

EPOCH_STEP_TRAIN = 2*12*NUM_TRAIN // BATCH_SIZE_TRAIN
EPOCH_STEP_TEST = 2*NUM_TEST // BATCH_SIZE_TEST 

UNet_model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
                      #tf.keras.metrics.MeanIoU(num_classes = 2),
                      'accuracy'])

NUM_OF_EPOCHS = 20
UNet_model.fit_generator(generator=train_generator, 
                    steps_per_epoch=EPOCH_STEP_TRAIN, 
                    validation_data=test_generator, 
                    validation_steps=EPOCH_STEP_TEST,
                    epochs=NUM_OF_EPOCHS,
                    callbacks=[early_stopping, tf.keras.callbacks.LearningRateScheduler(lr_exp_decay, verbose=1)]
                    )

#Evaluating the UNet models on the test partition
UNet_model.evaluate(test_generator, batch_size=128, steps=EPOCH_STEP_TEST)

#Saving the UNet model models with different depth levels and batch norm()
UNet_model.save('/home/idu/Desktop/COV19D/ChatGPT-saved-models/Segmentation models/UNet_model-3L-BatchNorm.h5')

#Loading saved models
UNet_model = keras.models.load_model('/home/idu/Desktop/COV19D/ChatGPT-saved-models/Segmentation models/UNet_model-3L-BatchNorm.h5')

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

import cv2 as cv      
  ## Segmenting images based on k-means clustering and exctracting lung regions and saving them 
#in the same directory as the original images (in .png format)
  
path = '/home/idu/Desktop/COV19D/segmentation/Segmentation Results/'
kernel = np.ones((5,5),np.float32)

def show_prediction(datagen, num=1):
    for i in range(0,num):
        image,mask = next(datagen)
        pred_mask = UNet_model.predict(image)[0] > 0.5
        print(pred_mask.dtype)
        #pred_mask = pred_mask.astype(np.float32)
        #pred_mask = pred_mask*255
        print(pred_mask.dtype)
        #pred_mask = pred_mask[None] 
        #pre_mask = np.squeeze(pred_mask, axis=0)
        #pred_mask = np.expand_dims(pred_mask, axis=0)
        print(pred_mask.dtype)
        #pred_mask = cv.dilate(pred_mask, kernel, iterations = 1)
        
        #pred_mask = Image.fromarray(pred_mask, 'L')
        #pred_mask = pred_mask.convert('LA')
        #pred_mask = np.expand_dims(pred_mask, axis=-1)
        #pred_mask.show()
        display([image[0], mask[0], pred_mask])        
        #num_classes = 2
        #IOU_keras = MeanIoU(num_classes=num_classes)
        #IOU_keras.update_state(mask[0], pred_mask)
        #print("Mean IoU =", IOU_keras.result().numpy())
        #mean_iou1.append(IOU_keras.result().numpy())

        #values = np.array(IOU_keras.get_weights()).reshape(num_classes, num_classes)
        #print(values) 
        

show_prediction(test_generator, 2)  


# calculating average, min and max dice coeffecient on the test set
GT_path = '/home/idu/Desktop/COV19D/segmentation/slices/test/mask/img'   
pred_path = '/home/idu/Desktop/COV19D/segmentation/slices/test/img/img'

# Mean IoU & Dice COeffecient Measures


# specify the img directory path
#path = "path/to/img/folder/"

# list files in img directory
files = os.listdir(GT_path)
files2 = os.listdir(pred_path)
print(files)

from tensorflow.keras import backend as K


def dice_coef(img, img2):
        if img.shape != img2.shape:
            raise ValueError("Shape mismatch: img and img2 must have to be of the same shape.")
        else:
            
            lenIntersection=0
            
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if ( np.array_equal(img[i][j],img2[i][j]) ):
                        lenIntersection+=1
             
            lenimg=img.shape[0]*img.shape[1]
            lenimg2=img2.shape[0]*img2.shape[1]  
            value = (2. * lenIntersection  / (lenimg + lenimg2))
        return value
    
mean_iou = []
dicee = []
dim = (224,224)
num_classes = 2
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
          #imgg = img
          
          #img = img > 0.5
          #img = numpy.bool(img)
          #img = np.asarray(img).astype(np.bool)

          ## Predicted mask
          p2 = os.path.join(pred_path, filee)
          img2 = cv2.imread(p2, 0)
          img2= cv2.resize(img2, dim)
          img2 = img2 / 255.0
          img2 = img2[None]
          img2 = np.expand_dims(img2, axis=-1)
          img2 = UNet_model.predict(img2) > 0.5
          #imgg2 = UNet_model.predict(img2) #> 0.5
          #imgg2 = imgg2.astype(numpy.float64)
          img2 = np.squeeze(img2)
          #imgg2 = np.squeeze(imgg2)
          #d = dtype(image)
          #print(d)
          #img2 = np.asarray(img2).astype(np.bool)
          
          IOU_keras = MeanIoU(num_classes=num_classes)
          IOU_keras.update_state(img, img2)
          print("Mean IoU =", IOU_keras.result().numpy())
          mean_iou.append(IOU_keras.result().numpy())

          value = dice_coef(img, img2)
          print("Dice coeffecient value is", value, "\n") 
          dicee.append(value)


UNet_model.summary()
#print (img)
#print(img2)
          
dicee = np.array(dicee)

L = len(mean_iou)
print("Number of Values is", L)

# Taking average of dice values
av=np.mean(mean_iou)
avv=np.mean(dicee)
print ("average value is", av)
print ("average value is", avv)

# Taking maximuim and minimuim of dice values
mx=np.max(mean_iou)
mxx=np.max(dicee)
print ("maximuim value is", mx)
print ("maximuim value is", mxx)

mn=np.min(mean_iou)
mnn=np.min(dicee)
print ("minimuim value is", mn)
print ("minimuim value is", mnn)

md=np.median(mean_iou)
mdd=np.median(dicee)
print ("median value is", md)
print ("median value is", mdd)






#############################################################
########################  Stage 2 : LUNG EXCTRACTION
##################################################################################


#UNet_model = tf.keras.models.load_model('/home/idu/Desktop/COV19D/segmentation/UNet_model.h5')

## Comparing the results of predicted masks between public dataset and COV19-CT database


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

# Fill in the small holes inside the binary mask of lungs
edges = roberts(binary)
binary = ndi.binary_fill_holes(edges)
plt.imshow(binary)

# Superimposing the binary image on the original image
#binary = int(binary)
binary=binary.astype(np.uint8)
final = cv2.bitwise_and(n1, n1, mask=binary)
plt.imshow(final)

#h = 255
#w = 298
dim = (224, 224)
dim = (h, w)

#kernel = np.ones((5, 5), np.uint8)

### Exctracting for all CT image in COV19-CT-DB
folder_path = '/home/idu/Desktop/COV19D/validation/non-covid' # Changoe this directory to loop over all training, validation and testing images
directory = '/home/idu/Desktop/COV19D/val-seg/non-covid'  # Changoe this directory to save the lung segmented images in the appropriate bath syncronizing with line above
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
            

################### Slice Removal After Lung Exctraction (Optional)


file_path1 = '/home/idu/Desktop/COV19D/train-seg/non-covid/ct_scan931/0.jpg'
file_path2 = '/home/idu/Desktop/COV19D/train-seg/non-covid/ct_scan931/40.jpg'
file_path3 = '/home/idu/Desktop/COV19D/train-seg/non-covid/ct_scan931/380.jpg'
file_path4 = '/home/idu/Desktop/COV19D/train-seg/non-covid/ct_scan931/420.jpg'
file_path5 = '/home/idu/Desktop/COV19D//train-seg/non-covid/ct_scan165/185.jpg'

n1 = cv2.imread(file_path1, 0)
n11 = n1.astype(float)
n11 /= 255.0 # Normallization
n_zeros = np.count_nonzero(n11==0)
n_zeros

n2 = cv2.imread(file_path2, 0)
n22 = n1.astype(float)
n22 /= 255.0 # Normallization
n_zeros = np.count_nonzero(n22==0)
n_zeros

n3 = cv2.imread(file_path3, 0)
n33 = n1.astype(float)
n33 /= 255.0 # Normallization
n_zeros = np.count_nonzero(n33)
n_zeros

n4 = cv2.imread(file_path4, 0)
n4 /= 255.0 # Normallization
n5 = cv2.imread(file_path5, 0)
n5 /= 255.0 # Normallization

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

histr = cv2.calcHist([n33],[0],None,[256],[0,256])
histr = np.histogram(n33)
plt.plot(histr)
plt.show()
hist,bins = np.histogram(n1.ravel(),256,[0,256])
plt.hist(n1.ravel(),256,[0,256])
plt.show()
plt.plot(hist, color = 'b')

hist,bins = np.histogram(n2.flatten(),256,[0,256])
plt.plot(hist, color = 'b')

hist,bins = np.histogram(n3.flatten(),256,[0,256])
plt.plot(hist, color = 'b')
#image2 = cv2.equalizeHist(image2)  ### Histogram equalization

# None-representative 
## [Uppermost]
count1 = np.count_nonzero(n11)
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
cv2.imwrite(output_path, lung_extracted_image)


################# Slice Removal Using ChatGPT [optional-Recommended]


def check_valid_image(image):
    return cv2.countNonZero(image) > 1764  # Choose a threshold for Removal

input_folder = "/home/idu/Desktop/COV19D/train-seg/covid"
output_folder = "/home/idu/Desktop/COV19D/train-seg-sliceremove/covid"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for subdir, dirs, files in os.walk(input_folder):
    subfolder_name = subdir.split('/')[-1]
    subfolder_path = os.path.join(output_folder, subfolder_name)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    count = 0
    for file in files:
        image_path = os.path.join(subdir, file)
        if '.jpg' in image_path:
            image = cv2.imread(image_path, 0)
            if check_valid_image(image):
                count += 1
                output_path = os.path.join(subfolder_path, file)
                cv2.imwrite(output_path, image)
    if count == 0:
        print(f"No valid images were found in subfolder: {subfolder_name}")


################################# Slice Cropping [optional]

img = cv2.imread('/home/idu/Desktop/COV19D/train-seg-removal/non-covid/ct_scan882/117.jpg')
img = skimage.color.rgb2gray(img)
r = cv2.selectROI(img)


count = []
folder_path = '/home/idu/Desktop/COV19D/train-seg-removal-crop/non-covid' 
#Change this directory to the directory where you need to do preprocessing for images
#Inside the directory must folder(s), which have the images inside them
for fldr in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, fldr)
        for filee in os.listdir(sub_folder_path):
            file_path = os.path.join(sub_folder_path, filee)
            img = cv2.imread(file_path)
            #Grayscale images
            img = skimage.color.rgb2gray(img) 
            # First cropping an image
            #%r = cv2.selectROI(im) 
            #Select ROI from images before you start the code 
            #Reference: https://learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/
            #{Last access 15th of Dec, 2021}
            # Crop image using r
            img_cropped = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            # Thresholding and binarizing images
            # Reference: https://datacarpentry.org/image-processing/07-thresholding/
            #{Last access 15th of Dec, 2021}
            # Gussian Filtering
            #img = skimage.filters.gaussian(img_cropped, sigma=1.0)
            # Binarizing the image         
            # Replace images with the image that includes ROI
            img_cropped = np.expand_dims(img_cropped, axis=2)
            img_cropped = array_to_img (img_cropped)
            img_cropped.save(str(file_path), 'JPEG')
             #print('saved')






#############################################################
########################  Stage 3 : CLASSIFICAIOTN 
##################################################################################



# Using imagedatagenerator

batch_size = 128

#h= 224
#w=224

#w = 152 # After cropping
#h = 104 # After cropping
train_datagen = ImageDataGenerator(rescale=1./255, 
                              vertical_flip=True,
                              horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/train-seg/',  ## COV19-CT-DB Training set
        target_size=(h, w),
        batch_size=batch_size,
        color_mode='grayscale',
        classes = ['covid','non-covid'],
        class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/val-seg/',  ## COV19-CT-DB Validation set
        target_size=(h, w),
        batch_size=batch_size,
        color_mode='grayscale',
        classes = ['covid','non-covid'],
        class_mode='binary')

#################### Transfer Learnign Classificaiton Approach [optional]

# Images must be 3 w
Model_Xcep = tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(h, w, 3))

for layer in Model_Xcep.layers:
	layer.trainable = False


model = tf.keras.Sequential([
    Model_Xcep, 
    tf.keras.layers.GlobalAveragePooling2D(), 
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()


h = 224
w = 224

h=w=512

##################### CNN model Classidier Approach

#from tensorflow.keras import models, layers, regularizers

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
    model.add(layers.Dropout(0.3))
    
    # Dense Layer  
    model.add(layers.Dense(1, activation='sigmoid'))
    
    
    return model

model = make_model()


### K-means Clustering Segmetnation + CNN - No slice removal
model.load_weights('/home/idu/Desktop/COV19D/ChatGPT-saved-models/kmeans-cluster-seg-cnn-classif.h5')

### UNet Seg + CNN - No slice removal
## Same as the Previous Architecture 
model.load_weights('/home/idu/Desktop/COV19D/ChatGPT-saved-models/UNet-BatchNorm-CNN-model.h5')

### UNet Seg + CNN - with slice removal

def make_model():
    model = models.Sequential()
    
    # Convulotional Layer 1
    model.add(layers.Conv2D(16, (3, 3), input_shape=(h, w, 1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convulotional Layer 2
    model.add(layers.Conv2D(32, (3, 3), padding="same"))  
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convulotional Layer 3
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())   
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convulotional Layer 4
    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convulotional Layer 5
    model.add(layers.Conv2D(256, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.3))
    
    # Dense Layer  
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

model = make_model()

model.load_weights('/home/idu/Desktop/COV19D/ChatGPT-saved-models/UNet-seg-sliceremove-cnn-class.h5')

## Choosing number of epoches
n_epochs= 100

###Learning Rate decay
def decayed_learning_rate(step):
  return initial_learning_rate * decay_rate ^ (step / decay_steps)

# Compiling the model 

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
                                  'accuracy'])


initial_learning_rate = 0.1
def lr_exp_decay(epoch, lr):
    k = 1
    return initial_learning_rate * math.exp(-k*epoch)


# early stopping
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)


initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

# saving weights
checkpoint = ModelCheckpoint('/home/idu/Desktop/COV19D/ChatGPT-saved-models/UNet-seg-5Layer-cnn-class.h5', save_best_only=True, save_weights_only=True)

# Class weight
counter = Counter(train_generator.classes)                          
max_val = float(max(counter.values()))       
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}  
class_weights

#training_steps = 2*269309 // batch_size
#training_steps = 434726 // batch_size ## Without Slice reduction
#val_steps = 67285 // batch_size
#val_steps = 106378 // batch_size ## Without Slice reduction


history=model.fit(train_generator,
                  #steps_per_epoch=training_steps,
                  validation_data=val_generator,
                  #validation_steps=val_steps,
                  verbose=2,
                  epochs=n_epochs,
                  callbacks=[early_stopping_cb, checkpoint],
                  class_weight=class_weights)


                  

model.evaluate(val_generator, batch_size=128)

##Evaluating the CNN model
print (history.history.keys())
            
Train_accuracy = history.history['accuracy']
print(Train_accuracy)
print(np.mean(Train_accuracy))
val_accuracy = history.history['val_accuracy']
print(val_accuracy)
print( np.mean(val_accuracy))

val_loss = history.history['val_loss']
print(val_loss)
print( np.mean(val_loss))

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


#############################################################################
###############################Making Prediciotns

## Input images shouldbe resized to 224x224 if any error
## Choosing the directory where the test/validation data is at
folder_path = '/home/idu/Desktop/COV19D/val-seg/non-covid'
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
        c = cv2.resize (c, (224, 224))
        c = c / 255.0
        #c=img_to_array(c)
        c = np.expand_dims(c, axis=-1)
        c = c[None]
        
        #result = model.predict_proba(c) #Probability of 1 (non-covid)
        result = model.predict(c)
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
print(len(coviddddd)) # 0.4 Threshold
print(len(covidd6)) # 0.5 Threshold
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




####################################################################
######### Using a Hybrid method [optional]
####################################################

#Actuall Covid
fulllist = list1 + list1n

#Actual nonCovid
fulllistt = listt1 + listt1n

####### Actual Covid Loop [In the validation set]

list1 = covidd6
list1n = noncovidd6

list2 = covidd6 # For model 2
list2n = noncovidd6

list3 = covidd6 # For model 3 [unET sEG + cnn]
list3n = noncovidd6

listt3 = covidd
listt3n = noncovidd

listtt3 = covidd8
listtt3n = noncovidd8

list4 = covidd6 # For model 4 [UNet Seg+ SliceRemoval+cnn]
list4n = noncovidd6

list4a = coviddddd # For model 4 
list4an = noncoviddddd

covid = []
noncovid = []

#### Making the decision for each CT scan

# If two models decide the CT scan is covid, then it will be considerd covid. Else the CT scan is non-covid
for item in listt3:
    if item in list2 or item in list3 or item in list1:
        covid.append(item)
    else:
        noncovid.append(item)

for item in listt3n:
    if item in list2n or item in list3n or item in list1n:
        noncovid.append(item)
    else:
        covid.append(item)
        
for item in fulllist:
    if item in listt3 and item in listtt3:
      covid.append(item)
    elif item in listt3n and item in listtt3n:
      noncovid.append(item)
    elif item in list3n and item in list2n:
      noncovid.append(item)
    else:
      covid.append(item)        

# Correctly Classified                        
print(covid)
print(len(covid))

# misclassified 
print(noncovid)
print(len(noncovid))

print(len(noncovid)+len(covid))
print(len(fulllist))
      
#print(len(covid+noncovid))

import csv


csv_filename = '/home/idu/Desktop/s/listt11.csv'

with open(csv_filename) as f:
    reader = csv.reader(f)
    listt11 = list(reader)



fulllistn = list11 + list11n
######### Actual non-Covid Loop [In the validation set]

list11 = covidd6
list11n = noncovidd6

list22 = covidd6 # For model 2
list22n = noncovidd6

list33 = covidd6 # For model 3 [UNet Seg + CNN]
list33n = noncovidd6

listt33 = covidd
listt33n = noncovidd

listtt33 = covidd8
listtt33n = noncovidd8


list44 = covidd6 # For model 4 [UNet seg - sliceremova+cnn]
list44n = noncovidd6

list44a = coviddddd # For model 4
list44an = noncoviddddd

covid2 = []
noncovid2 = []

## If two models decide the CT scan is covid, then it will be considerd covid. Else the CT scan is non-covid

for item in listt33:
    if item in list22 or item in list33 or item in list11:
        covid2.append(item)
    else:
        noncovid2.append(item)

for item in listt33n:
    if item in list22n or item in list33n or item in list11n:
        noncovid2.append(item)
    else:
        covid2.append(item)

fulllistt = listt33 + listt33n

for item in fulllistt:
    if item in listt33 and item in listtt33:
      covid2.append(item)
    elif item in listt33n and item in listtt33n:
      noncovid2.append(item)
    elif item in list33n and item in list22n:
      noncovid2.append(item)
    else:
      covid2.append(item) 
        
# Misclassified                   
print(covid2)
print(len(covid2))

# Correctly Classified
print(noncovid2)
print(len(noncovid2))
      
print(len(covid2)+len(noncovid2))


#####################################################################################
######### Saving to csv files format to report the results
###############################################
## Using Majority Voting for each CT scan



####0.5 slice level class probability 
with open('/home/idu/Desktop/s/listt11.csv', 'w') as f:
 wr = csv.writer(f, delimiter="\n")
 wr.writerow(listt11)

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
 
 ### KENAN MORANI - END OF THE CODE
 ##### github.com/kenanmorani