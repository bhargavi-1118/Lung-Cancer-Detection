#loading python require packages
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import pickle
import cv2
import os


def getUnetModel(input_size=(128,128,1)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), dilation_rate=2, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv1) #adding dilation rate for all layers
    conv1 = Dropout(0.1) (conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
 
    conv2 = Conv2D(64, (3, 3), dilation_rate=2, activation='relu', kernel_initializer='he_normal', padding='same') (pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv2)
    conv2 = Dropout(0.1) (conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
 
    conv3 = Conv2D(128, (3, 3), dilation_rate=2, activation='relu', padding='same')(pool2)#adding dilation to all layers
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), dilation_rate=2, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), dilation_rate=2, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), dilation_rate=2, activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), dilation_rate=2, activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), dilation_rate=2, activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), dilation_rate=2, activation='relu', padding='same')(up9)#adding dilation
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)#not adding dilation to last layer

    return Model(inputs=[inputs], outputs=[conv10])

X = []
Y = []
label = []

def loadDataset():
    global X, Y, label
    if os.path.exists("model/X.npy"):
        X = np.load("model/X.npy")
        Y = np.load("model/Y.npy")
        label = np.load("model/label.npy")
    else:
        for root, dirs, directory in os.walk("lidc-idri/image"):
            for j in range(len(directory)):
                name = directory[j]
                name = name.replace("img", "mask")
                if os.path.exists("lidc-idri/mask/"+name):
                    img = cv2.imread("lidc-idri/image/"+directory[j])
                    img = cv2.resize(img,(128, 128), interpolation = cv2.INTER_CUBIC)
                    X.append(img)
                    img = cv2.imread("lidc-idri/mask/"+name,0)
                    img = cv2.resize(img,(128, 128), interpolation = cv2.INTER_CUBIC)
                    Y.append(img)
                    white_pixels = np.sum(img == 255)
                    if white_pixels == 0:
                        label.append(0)
                    else:
                        label.append(1)
                    print(name+" "+directory[j]+" "+str(white_pixels))
        X = np.asarray(X)
        Y = np.asarray(Y)
        label = np.asarray(label)
        np.save("model/X",X)
        np.save("model/Y",Y)
        np.save("model/label",label)

loadDataset()
print(X.shape)
print(Y.shape)
print(label.shape)
print(label)
print(np.unique(label, return_counts=True))
'''
def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

cnn_model = getUnetModel(input_size=(128, 128, 1))
cnn_model.compile(optimizer=Adam(learning_rate=1e-4), loss=[dice_coef_loss], metrics = [dice_coef, 'binary_accuracy']) #compiling model
cnn_model.load_weights("model/unet_weights.hdf5")
for root, dirs, directory in os.walk("lidc-idri/image"):
    for j in range(len(directory)):
        print(directory[j])
        img = cv2.imread("lidc-idri/image/"+directory[j],0)
        image = img
        img = cv2.resize(img,(128, 128), interpolation = cv2.INTER_CUBIC)
        img = (img-127.0)/127.0
        img = img.reshape(1,128,128,1)
        preds = cnn_model.predict(img)#predict segmented image
        preds = preds[0]
        #cv2.imshow("original", image)
        cv2.imshow("mask12", preds*255)
        cv2.waitKey(0)
        
