import os
import matplotlib
matplotlib.use('Agg')
import csv
import pydicom
import numpy as np
import pickle
from morph_layers2D import Dilation2D, Erosion2D

import keras
import random
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from skimage import measure
from skimage.transform import resize
from skimage import transform, color

from matplotlib import pyplot as plt
import matplotlib.patches as patches

from keras.models import load_model
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from attention_module import attach_attention_module

# 1. pre-defined variables
batch_size = 32           # batch size
height, width = 128 ,128  # input size
n_class = 2               # number of class

# 2. data
pneumonia_locations = {}
pneumonia_target = {}
with open(os.path.join('/data/shaobo/input/NewLabel1.csv'), mode='r') as infile:
    #with open(os.path.join('/data/shaobo/input/NewLabel1.csv'), mode='r') as infile:
    reader = csv.reader(infile)
    next(reader, None)
    
    count_1, count_0 = 0, 0
    for rows in reader:
        filename = rows[0]
        location = rows[1:5]
        target = rows[5]
        
        #target = [int(float(i)) for i in target]
        if target == '1':
            #if count_1 < 6000:
            location = [int(float(i)) for i in location]
            if filename in pneumonia_locations:
                pneumonia_locations[filename].append(location)
            else:
                pneumonia_locations[filename] = [location]
                count_1 += 1
            #else:
            #    continue
        if target == '0':
            if count_0 < 6000:
                pneumonia_locations[filename] = [[0,0,0,0]]
                count_0 += 1
            else:
                continue
        pneumonia_target[filename] = keras.utils.to_categorical(target, n_class)



path_here = '/data/shaobo/input/stage_1_train_images/'
# filenames = os.listdir(path_here)
folder = list(pneumonia_locations.keys())
random.shuffle(folder)
n_valid_samples = 1000
#n_test_samples = 500
train_filenames = folder[n_valid_samples +1:]
valid_filenames = folder[0:n_valid_samples]
#test_filenames = folder[n_valid_samples:n_valid_samples + n_test_samples]
data_len = len(folder)
print('training with [',data_len, '] images.')
print('with [', count_1, '] pneumonia.','And [', count_0, '] Healthy')
print('validation with[',len(valid_filenames),'] images.')

def get_batch(folder,batch_size):
    while True:
        c = np.random.choice(folder, batch_size*4)
        
        count_batch = 0
        
        img_in_all = []
        img_seg_gt_all = []
        img_target_all = []
        for each_file in c:
            
            each_file_with_ex = each_file + '.dcm'
            
            try:
                img_in = pydicom.dcmread(os.path.join(path_here, each_file_with_ex)).pixel_array
            except:
                # del c[index(each_file)]
                continue
                # img_in = np.zeros((height, width))
            
            img_seg_gt = np.zeros(img_in.shape)
            for location in pneumonia_locations[each_file]:
                if sum(location) != 0:
                    x, y, w, h = location
                    img_seg_gt[y:y+h, x:x+w] = 1
                    
            img_in = rgb2gray(img_in)
            img_in = transform.resize(img_in, (height, width, 1), mode='reflect')
            img_seg_gt = transform.resize(img_seg_gt, (height, width, 1), mode='reflect')
            img_target = pneumonia_target[each_file]

            img_in_all.append(img_in)
            img_seg_gt_all.append(img_seg_gt)
            img_target_all.append(img_target)
            
            if count_batch >= batch_size-1:
                break
            count_batch += 1
            
        img_in_all = np.array(img_in_all)
        img_in_all = np.reshape(img_in_all, [batch_size, height, width, 1])
        img_seg_gt_all = np.array(img_seg_gt_all)
        img_seg_gt_all = np.reshape(img_seg_gt_all, [batch_size, height, width, 1])
        img_target_all = np.array(img_target_all)
        img_target_all = np.reshape(img_target_all, [batch_size, n_class])
        
        #plt.imshow(np.reshape(img_in_all[0],[height, width]), cmap='gray')
        #plt.show()
        #plt.imshow(np.reshape(img_seg_gt_all[0],[height, width]), cmap='gray')
        #plt.show()
        #print(type(img_target_all),img_target_all[0],img_target_all.shape,img_target_all)
        #return
        yield ({'image_in': img_in_all}, \
              {'segmentation': img_seg_gt_all, 'classification': img_target_all})

### layer / model
from keras.layers import Input, Conv2D, concatenate, add, Dense, Dropout, MaxPooling2D, Flatten, \
                          UpSampling2D, Reshape, BatchNormalization, LeakyReLU, MaxPool2D, AveragePooling2D
from keras.models import Model
import keras.backend as K
from keras.utils import multi_gpu_model
from keras.utils import plot_model
#from keras.utils.vis_utils import plot_model


def Vgg16CBAM(in_image=(height, width, 1)):

    img_in = Input(shape = in_image, name='image_in')
    img_in_b = BatchNormalization(name='in_BN')(img_in)
    
    
    #c0 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='c0')(img_in_b)
    #c1 = Dilation2D(8, (3,3),padding="same",strides=(1,1))(img_in_b)
    #c1 = Erosion2D(8, (3,3),padding="same",strides=(1,1))(c1)
    #c1 = BatchNormalization()(c1)
    
    #c1 = Erosion2D(8, (3,3),padding="same",strides=(1,1))(c1)
    #c1 = Dilation2D(8, (3,3),padding="same",strides=(1,1))(c1)
    #c1 = BatchNormalization()(c1)
    #c1 = attach_attention_module(c1, attention_module='cbam_block')
    #c10 = add([c0,c1])
    
    ## cc00 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='cc0')(cc0)
    #cc1 = Erosion2D(8, (3,3),padding="same",strides=(1,1))(cc0)
    #cc1 = Dilation2D(8, (3,3),padding="same",strides=(1,1))(cc1)
    #cc1 = BatchNormalization()(cc1)
    #cc1 = attach_attention_module(cc1, attention_module='cbam_block')
    #cc10 = add([cc00,cc1])
    
    
    conv0 = Conv2D(32, 6, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv0_5')(img_in_b)
    conv0 = Conv2D(32, 6, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv0_6')(conv0)
    conv0 = BatchNormalization()(conv0)
    #conv0 = attach_attention_module(conv0, attention_module='cbam_block')
    pool0 = MaxPooling2D(pool_size=(2, 2),name='down0')(conv0)
    
    
    conv1 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv1_1')(pool0)
    conv1 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv1_2')(conv1)
    conv1 = BatchNormalization()(conv1)
    #conv1 = attach_attention_module(conv1, attention_module='cbam_block')
    pool1 = MaxPooling2D(pool_size=(2, 2),name='down1')(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv2_1' )(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv2_2')(conv2)
    conv2 = BatchNormalization()(conv2)
   # conv2 = attach_attention_module(conv2, attention_module='cbam_block')
    pool2 = MaxPooling2D(pool_size=(2, 2),name='down2')(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv3_1')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv3_2')(conv3)
    conv3 = BatchNormalization()(conv3)
    #conv3 = attach_attention_module(conv3, attention_module='cbam_block')
    pool3 = MaxPooling2D(pool_size=(2, 2),name='down3')(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv4_1')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv4_2')(conv4)
    conv4 = BatchNormalization()(conv4)
    #conv4 = attach_attention_module(conv4, attention_module='cbam_block')
    pool4 = MaxPooling2D(pool_size=(2, 2),name='down4')(conv4)

    down_4_f = Flatten(name='down_2_flat')(pool4)

    down_classsify = Dense(512,activation='relu',name='classify_1')(down_4_f)
    down_classsify = Dropout(0.6)(down_classsify)
    down_classsify = Dense(128,activation='relu',name='classify_2')(down_classsify)
    down_classsify = Dropout(0.65)(down_classsify)
    classification = Dense(2,activation='sigmoid',name='classification')(down_classsify)



    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv5_1')(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv5_2')(conv5)
    conv5 = BatchNormalization()(conv5)
  

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv6_1')(conv5)
    up6 = UpSampling2D(size = (2,2),name = 'up_1')(up6)
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv6_2')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name ='conv6_3')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv7_1')(conv6)
    up7 = UpSampling2D(size = (2,2),name = 'up2')(up7)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv7_2')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv7_3')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv8_1')(conv7)
    up8 = UpSampling2D(size = (2,2),name ='up3')(up8)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv8_2')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv8_3')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv9_1')(conv8)
    up9 = UpSampling2D(size = (2,2),name = 'up4')(up9)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv9_2')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv9_3')(conv9)
    conv9 = BatchNormalization()(conv9)
    
    up10 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv10_1')(conv9)
    up10 = UpSampling2D(size = (2,2),name = 'up5')(up10)
    merge10 = concatenate([conv0,up10], axis = 3)
    conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv10_2')(merge10)
    conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv10_3')(conv10)
    conv10 = BatchNormalization()(conv10)
    
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv10_4')(conv10)

    segmentation = Conv2D(1, 1, activation = 'sigmoid', name='segmentation')(conv10)

    model = Model(inputs = img_in, outputs = [segmentation, classification])
    model.summary()


    plot_model(model, to_file='model.png')
    return model

   

def unet(in_image=(height, width, 1)):
    img_in = Input(shape = in_image, name='image_in')
    # image preprocessing module
    img_in_B = BatchNormalization(name='in_BN')(img_in)
    
    #c0 = img_in
    #c0 = Conv2D(12, 8, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='datapre_0')(c0)
    
    #c1 = Dilation2D(6, (6,6),padding="same",strides=(1,1))(img_in)
    #c1 = Erosion2D(6, (6,6),padding="same",strides=(1,1))(c1)
    
    #c2 = Erosion2D(6, (6,6),padding="same",strides=(1,1))(img_in)
    #c2 = Dilation2D(6, (6,6),padding="same",strides=(1,1))(c2)
    
    #c12= concatenate([c1,c2], axis = 3) #12 filters
    
    #mb1 = add([c0, c12])
    
    #imgpre = Conv2D(6, 6, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='datapre_1')(mb1)
    
    
    
    
    
    cc1 = Erosion2D(6, (6,6),padding="same",strides=(1,1))(img_in_B)
    cc1 = Dilation2D(6, (6,6),padding="same",strides=(1,1))(cc1)
    #cc12= concatenate([cc1,cc2], axis = 3) #12 filters
    
    #imgpre2 = Conv2D(6, 6, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='datapre_2')(cc12)
    
    #ccc1 =  Dilation2D(6, (6,6),padding="same",strides=(1,1))(imgpre2)
   # ccc2 = Erosion2D(6, (6,6),padding="same",strides=(1,1))(imgpre2)
    #ccc12= concatenate([ccc1,ccc2], axis = 3) #12 filters
  
    
    #c3 = Dilation2D(6, (6,6),padding="same",strides=(1,1))(imgpre)
   # c3 = Dilation2D(6, (6,6),padding="same",strides=(1,1))(c3)
    
    #c4 = Erosion2D(6, (6,6),padding="same",strides=(1,1))(c3)
   # c4 = Erosion2D(6, (6,6),padding="same",strides=(1,1))(c4)
    
    
    #c5 = Conv2D(6, 6, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv0_2')(c4)
    
    #c6 = Dilation2D(6, (6,6),padding="same",strides=(1,1))(imgpre2)
   #c6 = Dilation2D(6, (6,6),padding="same",strides=(1,1))(c6)
    
    #c7 = Erosion2D(6, (6,6),padding="same",strides=(1,1))(imgpre2)
    #c7 = Erosion2D(6, (6,6),padding="same",strides=(1,1))(c7)
    #c8 = concatenate([c6,c7], axis = 3)
    
    conv0 = Conv2D(32, 6, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv0_5')(cc1)
    conv0 = Conv2D(32, 6, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv0_6')(conv0)
    conv0 = BatchNormalization()(conv0)
    pool0 = MaxPooling2D(pool_size=(2, 2),name='down0')(conv0)
    
    
    conv1 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv1_1')(pool0)
    conv1 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv1_2')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),name='down1')(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv2_1' )(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv2_2')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),name='down2')(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv3_1')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv3_2')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),name='down3')(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv4_1')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv4_2')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),name='down4')(conv4)

    down_4_f = Flatten(name='down_2_flat')(pool4)

    down_classsify = Dense(512,activation='relu',name='classify_1')(down_4_f)
    down_classsify = Dropout(0.6)(down_classsify)
    down_classsify = Dense(128,activation='relu',name='classify_2')(down_classsify)
    down_classsify = Dropout(0.65)(down_classsify)
    classification = Dense(2,activation='sigmoid',name='classification')(down_classsify)



    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv5_1')(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv5_2')(conv5)
    conv5 = BatchNormalization()(conv5)
  

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv6_1')(conv5)
    up6 = UpSampling2D(size = (2,2),name = 'up_1')(up6)
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv6_2')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name ='conv6_3')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv7_1')(conv6)
    up7 = UpSampling2D(size = (2,2),name = 'up2')(up7)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv7_2')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv7_3')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv8_1')(conv7)
    up8 = UpSampling2D(size = (2,2),name ='up3')(up8)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv8_2')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv8_3')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv9_1')(conv8)
    up9 = UpSampling2D(size = (2,2),name = 'up4')(up9)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv9_2')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv9_3')(conv9)
    conv9 = BatchNormalization()(conv9)
    
    up10 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv10_1')(conv9)
    up10 = UpSampling2D(size = (2,2),name = 'up5')(up10)
    merge10 = concatenate([conv0,up10], axis = 3)
    conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv10_2')(merge10)
    conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv10_3')(conv10)
    conv10 = BatchNormalization()(conv10)
    
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv10_4')(conv10)

    segmentation = Conv2D(1, 1, activation = 'sigmoid', name='segmentation')(conv10)

    model = Model(inputs = img_in, outputs = [segmentation, classification])
    model.summary()

    return model

def unet1(in_image=(height, width, 1)):
  img_in = Input(shape = in_image, name='image_in')
  # image preprocessing module
  img_in_B = BatchNormalization(name='in_BN')(img_in)
  
  c0 = img_in
  c0 = Conv2D(12, 8, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv0_0')(c0)
  
  c1 = Dilation2D(6, (6,6),padding="same",strides=(1,1))(img_in)
  c1 = Erosion2D(6, (6,6),padding="same",strides=(1,1))(c1)
  
  c2 = Erosion2D(6, (6,6),padding="same",strides=(1,1))(img_in)
  c2 = Dilation2D(6, (6,6),padding="same",strides=(1,1))(c2)
  
  c12= concatenate([c1,c2], axis = 3) #10 filters
  
  mb1 = add([c0, c12])
  
  imgpre = Conv2D(6, 6, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv0_1')(mb1)
  
  c3 = Dilation2D(6, (6,6),padding="same",strides=(1,1))(imgpre)
  c3 = Dilation2D(6, (6,6),padding="same",strides=(1,1))(c3)
  
  c4 = Erosion2D(6, (6,6),padding="same",strides=(1,1))(c3)
  c4 = Erosion2D(6, (6,6),padding="same",strides=(1,1))(c4)
  
  
  c5 = Conv2D(6, 6, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv0_2')(c4)
  
  #c6 = Dilation2D(6, (6,6),padding="same",strides=(1,1))(imgpre2)
 #c6 = Dilation2D(6, (6,6),padding="same",strides=(1,1))(c6)
  
  #c7 = Erosion2D(6, (6,6),padding="same",strides=(1,1))(imgpre2)
  #c7 = Erosion2D(6, (6,6),padding="same",strides=(1,1))(c7)
  #c8 = concatenate([c6,c7], axis = 3)
  
  conv0 = Conv2D(32, 6, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv0_5')(c5)
  conv0 = Conv2D(32, 6, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv0_6')(conv0)
  conv0 = BatchNormalization()(conv0)
  pool0 = MaxPooling2D(pool_size=(2, 2),name='down0')(conv0)
  
  
  conv1 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv1_1')(pool0)
  conv1 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv1_2')(conv1)
  conv1 = BatchNormalization()(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2),name='down1')(conv1)
  
  conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv2_1' )(pool1)
  conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv2_2')(conv2)
  conv2 = BatchNormalization()(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2),name='down2')(conv2)
  
  conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv3_1')(pool2)
  conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv3_2')(conv3)
  conv3 = BatchNormalization()(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2),name='down3')(conv3)
  
  conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv4_1')(pool3)
  conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv4_2')(conv4)
  conv4 = BatchNormalization()(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2),name='down4')(conv4)

  down_4_f = Flatten(name='down_2_flat')(pool4)

  down_classsify = Dense(512,activation='relu',name='classify_1')(down_4_f)
  down_classsify = Dropout(0.6)(down_classsify)
  down_classsify = Dense(128,activation='relu',name='classify_2')(down_classsify)
  down_classsify = Dropout(0.65)(down_classsify)
  classification = Dense(2,activation='sigmoid',name='classification')(down_classsify)



  conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv5_1')(pool4)
  conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv5_2')(conv5)
  conv5 = BatchNormalization()(conv5)


  up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv6_1')(conv5)
  up6 = UpSampling2D(size = (2,2),name = 'up_1')(up6)
  merge6 = concatenate([conv4,up6], axis = 3)
  conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv6_2')(merge6)
  conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name ='conv6_3')(conv6)
  conv6 = BatchNormalization()(conv6)

  up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv7_1')(conv6)
  up7 = UpSampling2D(size = (2,2),name = 'up2')(up7)
  merge7 = concatenate([conv3,up7], axis = 3)
  conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv7_2')(merge7)
  conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv7_3')(conv7)
  conv7 = BatchNormalization()(conv7)

  up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv8_1')(conv7)
  up8 = UpSampling2D(size = (2,2),name ='up3')(up8)
  merge8 = concatenate([conv2,up8], axis = 3)
  conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv8_2')(merge8)
  conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv8_3')(conv8)
  conv8 = BatchNormalization()(conv8)

  up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv9_1')(conv8)
  up9 = UpSampling2D(size = (2,2),name = 'up4')(up9)
  merge9 = concatenate([conv1,up9], axis = 3)
  conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv9_2')(merge9)
  conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv9_3')(conv9)
  conv9 = BatchNormalization()(conv9)
  
  up10 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv10_1')(conv9)
  up10 = UpSampling2D(size = (2,2),name = 'up5')(up10)
  merge10 = concatenate([conv0,up10], axis = 3)
  conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv10_2')(merge10)
  conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'conv10_3')(conv10)
  conv10 = BatchNormalization()(conv10)
  
  conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv10_4')(conv10)

  segmentation = Conv2D(1, 1, activation = 'sigmoid', name='segmentation')(conv10)

  model = Model(inputs = img_in, outputs = [segmentation, classification])
  model.summary()

  return model
  
import keras.backend as K


def iou_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    return 1 - score

def seg_loss(y_true,y_pred):
        return 0.8 * keras.losses.binary_crossentropy(y_true, y_pred) + 0.2* iou_loss(y_true, y_pred)


def compile_model():
    
    opti = keras.optimizers.adadelta(lr=0.05)
    model = Vgg16CBAM()
    #model = ResCBAM1()
    model.compile(optimizer=opti,
                  loss=['binary_crossentropy', 'binary_crossentropy'],
                  loss_weights=[0.75, 0.25],
                  metrics = {'classification':'accuracy'})

    return model

def train(epoch=5):
    model = compile_model()

    history = model.fit_generator(get_batch(train_filenames,batch_size), validation_data = get_batch(valid_filenames,batch_size), \
                                 steps_per_epoch=int(data_len / batch_size), epochs=epoch, validation_steps= int(500 / batch_size))
    model.save('/home/CVL1/Shaobo/PneSeg/model2/RSNA_seg_Shaobo.h5')
    with open('/home/CVL1/Shaobo/PneSeg/model2/history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

from matplotlib import pyplot as plt
def plot_loss():
    with open('/home/CVL1/Shaobo/PneSeg/model2/history.pkl', 'rb') as file_pi:
        history = pickle.load(file_pi)

    plt.figure(figsize=(12, 4))
    plt.subplot(141)
    plt.plot(history["loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Valid loss")
    plt.legend()
    plt.subplot(142)
    plt.plot(history["segmentation_loss"], label="segmentation_loss")
    plt.plot(history["val_segmentation_loss"], label="val_segmentation_loss")
    #plt.plot(history["segmentation_iou_score"],label = "segmentation_iou_score")
    #plt.plot(history["val_segmentation_iou_score"],label = "val_segmentation_iou_score")
    plt.legend()
    plt.subplot(143)
    plt.plot(history["classification_loss"], label="classification_loss")
    plt.plot(history["val_classification_loss"], label="val_classification_loss")
    plt.legend()
    plt.subplot(144)
    plt.plot(history["classification_acc"],label = "classification_acc")
    plt.plot(history["val_classification_acc"],label = "val_classification_acc")
    plt.legend()
    plt.show()
    plt.savefig('/home/CVL1/Shaobo/PneSeg/model2/training.png')
    return

from keras.models import load_model
from skimage.filters import threshold_otsu
import vis
import vis.visualization
import vis.utils
from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam, overlay
from vis.utils import utils
from keras import activations
from skimage.transform import resize
import matplotlib.cm as cm
from keras.utils import CustomObjectScope

def plot_img():
    
    n_test_samples = 500
    with CustomObjectScope({'Dilation2D': Dilation2D, 'Erosion2D':Erosion2D, 'attach_attention_module': attach_attention_module}):
        model = load_model('/home/CVL1/Shaobo/PneSeg/model2/RSNA_seg_Shaobo.h5')
        
    gb = get_batch(valid_filenames,n_test_samples)
    abatch = next(gb)
    imgs = abatch[0]['image_in']
    msks = abatch[1]['segmentation']
    labels = abatch[1]['classification']

    f, axarr = plt.subplots(25, 20, figsize=(64,64))
    axarr = axarr.ravel()
    axidx = 0
    IOU_score = []
    precision_score = []
    axarr = axarr.ravel()
    count_Seg = []
    count_SegIoU = []
    count_class = []
    
    for i in range(500):
        img = imgs[i]
        msk = msks[i]
        class_label = labels[i]

        img1 = np.reshape(img,[height, width])
        axarr[axidx].imshow(img1)

        msk = np.reshape(msk,[height, width])
        comp = msk > 0.5 / 1.
        comp1 = measure.label(comp)
        predictionString = ''
        for region in measure.regionprops(comp1):
            # retrieve x, y, height and width
            y11, x11, y12, x12 = region.bbox
            height1 = y12 - y11
            width1 = x12 - x11
            axarr[axidx].add_patch(patches.Rectangle((x11,y11),width1,height1,linewidth=2,edgecolor='b',facecolor='none'))

    ### segment
        predImg = np.reshape(imgs[i],[1, height, width, 1])
        seg_pre, class_pre = model.predict(predImg)

        pred = np.reshape(seg_pre,[height, width])
        pred = (pred > 0.5) / 1.
        pred1 =  measure.label(pred)
        predictionString = ''
        for region in measure.regionprops(pred1):
            # retrieve x, y, height and width
            y21, x21, y22, x22 = region.bbox
            height2 = y22 - y21
            width2 = x22 - x21
            axarr[axidx].add_patch(patches.Rectangle((x21,y21),width2,height2,linewidth=2,edgecolor='r',facecolor='none'))
            
        axidx += 1
        intersection = np.logical_and(comp1, pred1)
        union = np.logical_or(comp1, pred1)
        iou_score =np.sum(intersection) / np.sum(union)
        precision =np.sum(intersection) / np.sum(comp1)

        if (np.isnan(precision) == True):
            precision=0
            count_Seg.append(0)
        else:
            count_Seg.append(1)
        precision_score.append(precision)

        if (np.isnan(iou_score) == True):
            iou_score=0
            count_SegIoU.append(0)
        else:
            count_SegIoU.append(1)
        IOU_score.append(iou_score)
    plt.show()
    plt.savefig('/home/CVL1/Shaobo/PneSeg/model2/plot_valid.png')
    print(IOU_score)
    print("mean_IoU",sum(IOU_score)/sum(count_SegIoU))

    print("mean_Precision",sum(precision_score)/sum(count_Seg))
   #print("Test_Accuracy",sum(count_class)/n_test_samples)
    #print("count_label",count_label)
    #print("count_pred",count_pred)
    return

def plot_saliency():
    with CustomObjectScope({'Dilation2D': Dilation2D, 'Erosion2D': Erosion2D, 'attach_attention_module': attach_attention_module}):
        model = load_model('/home/CVL1/Shaobo/PneSeg/model2/RSNA_seg_Shaobo.h5')
    model.summary()
    gb = get_batch(valid_filenames,250)
    abatch = next(gb)
    imgs = abatch[0]['image_in']
    msks = abatch[1]['segmentation']
    labels = abatch[1]['classification']
    plt.figure(figsize=(40,20))
    n = 1
    for i in range(n):
        img = imgs[i]
        
        msk = msks[i]
        class_label = labels[i]
        img1 = np.reshape(img,[height, width])
        msk = np.reshape(msk,[height, width])
        comp = msk > 0.5 / 1.
        comp1 = measure.label(comp)
        
        predImg = np.reshape(imgs[i],[1, height, width, 1])
        seg_pre, class_pre = model.predict(predImg)
        pred = np.reshape(seg_pre,[height, width])
        pred = (pred > 0.5) / 1.
        pred1 =  measure.label(pred)
        
        class_idx = np.where(class_label == 1.)[0]
        
        ax1 = plt.subplot(3, n, i+1)
        ax1.imshow(img1)
        ax1.set_title('Seg_Result '+ str(class_idx) )
        #ax1.get_xaxis().set_visible(False)
        #ax1.get_yaxis().set_visible(False)
        for region in measure.regionprops(comp1):
            # retrieve x, y, height and width
            y11, x11, y12, x12 = region.bbox
            height1 = y12 - y11
            width1 = x12 - x11
            ax1.add_patch(patches.Rectangle((x11,y11),width1,height1,linewidth=1,edgecolor='b',facecolor='none'))
        for region in measure.regionprops(pred1):
            # retrieve x, y, height and width
            y21, x21, y22, x22 = region.bbox
            height2 = y22 - y21
            width2 = x22 - x21
            ax1.add_patch(patches.Rectangle((x21,y21),width2,height2,linewidth=1,edgecolor='r',facecolor='none'))
    
    #class_idx = np.where(class_label == 1.)[0]
        print(class_label)
        print(class_idx[0])
        layer_idx = utils.find_layer_idx(model, 'classification')
        print(layer_idx)
        model.layers[layer_idx].activation = activations.linear
        model = utils.apply_modifications(model, custom_objects = {'Dilation2D': Dilation2D, 'Erosion2D': Erosion2D})
        img2 = resize(imgs[i],(height, width))
        
        grads = visualize_saliency(model, layer_idx, filter_indices= class_idx[0], seed_input = img2)
        #grads = visualize_cam(model1, layer_idx, filter_indices= class_idx[0], seed_input = img2, backprop_modifier='guided')
        ax2 = plt.subplot(3, n, i+1+n)
        ax2.imshow(grads,cmap='jet')
        ax2.set_title('Saliency Map')
        #ax4.get_xaxis().set_visible(False)
       # ax4.get_yaxis().set_visible(False)
        for region in measure.regionprops(comp1):
            y11, x11, y12, x12 = region.bbox
            height1 = y12 - y11
            width1 = x12 - x11
            ax2.add_patch(patches.Rectangle((x11,y11),width1,height1,linewidth=1,edgecolor='g',facecolor='none'))
        for region in measure.regionprops(pred1):
            # retrieve x, y, height and width
            y21, x21, y22, x22 = region.bbox
            height2 = y22 - y21
            width2 = x22 - x21
            ax2.add_patch(patches.Rectangle((x21,y21),width2,height2,linewidth=1,edgecolor='r',facecolor='none'))
            
        
          
           
        model.layers[layer_idx].activation = keras.activations.linear
        model = utils.apply_modifications(model, custom_objects = {'Dilation2D': Dilation2D, 'Erosion2D': Erosion2D})
        #print(layer_idx)
        #model.layers[layer_idx].activation = activations.linear
        penultimate_layer = utils.find_layer_idx(model, 'down4')
        print(penultimate_layer)
        #model = utils.apply_modifications(model)
        img2 = resize(imgs[i],(height, width))
           
        grads = visualize_cam(model, layer_idx, filter_indices= class_idx[0], seed_input = img2, penultimate_layer_idx = penultimate_layer)
        ax3 = plt.subplot(3, n, i+1+2*n)
        ax3.imshow(grads, cmap='jet', alpha=0.8)
        
        #ax4.set_title('vanilla')
        #ax4.get_xaxis().set_visible(False)
        #ax4.get_yaxis().set_visible(False)
        for region in measure.regionprops(comp1):
            y11, x11, y12, x12 = region.bbox
            height1 = y12 - y11
            width1 = x12 - x11
            ax3.add_patch(patches.Rectangle((x11,y11),width1,height1,linewidth=1,edgecolor='g',facecolor='none'))
        for region in measure.regionprops(pred1):
               # retrieve x, y, height and width
            y21, x21, y22, x22 = region.bbox
            height2 = y22 - y21
            width2 = x22 - x21
            ax3.add_patch(patches.Rectangle((x21,y21),width2,height2,linewidth=1,edgecolor='r',facecolor='none'))
               
                   
    plt.show()
    plt.savefig('/home/CVL1/Shaobo/PneSeg/model2/test.jpg')



if __name__ == "__main__":
    print("===============")
    #train(epoch=30)
   # plot_loss()
    #plot_img()
    plot_saliency()
    





