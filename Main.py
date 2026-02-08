from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter import simpledialog
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import cv2
import os
import numpy as np
#loading python require packages
import pandas as pd
import matplotlib.pyplot as plt
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
from keras.callbacks import ModelCheckpoint
import pickle
import seaborn as sns
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv3D, MaxPooling3D, LSTM,RepeatVector
from keras.utils import to_categorical
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


main = tkinter.Tk()
main.title("Pancreatic Tumor Detection using Image Processing") #designing main screen
main.geometry("1000x650")


global filename, X, Y
global X_train, X_test, y_train, y_test, unet_model, ccdc_model

def loadDataset():
    global X, Y, label
    if os.path.exists("model/X2.npy"):
        X = np.load("model/X2.npy")
        Y = np.load("model/Y2.npy")
    else:
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = directory[j]
                name = name.replace("img", "mask")
                if os.path.exists("lidc-idri/mask/"+name):
                    img = cv2.imread("lidc-idri/image/"+directory[j])
                    img = cv2.resize(img,(32, 32), interpolation = cv2.INTER_CUBIC)
                    img = img.astype('float32')
                    img = img/255
                    X.append([img])
                    img = cv2.imread("lidc-idri/mask/"+name,0)
                    img = cv2.resize(img,(128, 128), interpolation = cv2.INTER_CUBIC)
                    white_pixels = np.sum(img == 255)
                    if white_pixels == 0:
                        Y.append(0)
                    else:
                        Y.append(1)
                    print(name+" "+str(white_pixels))    
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save("model/X1",X)
        np.save("model/Y1",Y)

def uploadDataset():
    global filename, X, Y, labels
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    X = []
    Y = []
    loadDataset()
    text.insert(END,"Total images loaded = "+str(X.shape[0]))

def processDataset():
    global X, Y, labels
    text.delete('1.0', END)
    dim = 128
    img = X[0]
    img = img[0]
    print(img.shape)
    text.insert(END,"Dataset Processing & Normalization Complated")
    img = cv2.resize(img, (300, 300))
    cv2.putText(img, "Sample Processed Image", (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    cv2.imshow('Processed Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def trainTestSplit():
    global X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)#shuffle all images
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test    
    text.insert(END,"Dataset Training & Testing Details\n\n")
    text.insert(END,"80% images for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% images for testing  : "+str(X_test.shape[0])+"\n")

#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n")
    labels = ['Benign', 'Malignant']
    conf_matrix = confusion_matrix(testY, predict)
    fig, axs = plt.subplots(1,2,figsize=(10, 3))
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axs[0]);
    ax.set_ylim([0,len(labels)])
    axs[0].set_title(algorithm+" Confusion matrix") 

    random_probs = [0 for i in range(len(testY))]
    p_fpr, p_tpr, _ = roc_curve(testY, random_probs, pos_label=1)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='orange',label="True classes")
    ns_tpr, ns_fpr, _ = roc_curve(testY, predict, pos_label=1)
    axs[1].plot(ns_tpr, ns_fpr, linestyle='--', label='Predicted Classes')
    axs[1].set_title(algorithm+" ROC AUC Curve")
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive rate')
    plt.show()

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)    

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

def runCNN():
    text.delete('1.0', END)
    global unet_model, ccdc_model
    global X_train, X_test, y_train, y_test
    unet_model = getUnetModel(input_size=(128, 128, 1))
    unet_model.compile(optimizer=Adam(learning_rate=1e-4), loss=[dice_coef_loss], metrics = [dice_coef, 'binary_accuracy']) #compiling model
    unet_model.load_weights("model/unet_weights.hdf5")
    ccdc_model = Sequential()
    #creating CNN3d layer with 1 X 3 X 3 matrix to filtered features using 32 neurons
    ccdc_model.add(Conv3D(32, (1, 3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], X.shape[3], X.shape[4])))
    #max pool to collect relevant features from CNN3D layer
    ccdc_model.add(MaxPooling3D((1, 2, 2)))
    #adding another layer
    ccdc_model.add(Conv3D(16, (1, 3, 3), activation='relu'))
    ccdc_model.add(MaxPooling3D((1, 2, 2)))
    ccdc_model.add(Conv3D(16, (1, 3, 3), activation='relu'))
    ccdc_model.add(MaxPooling3D((1, 2, 2)))
    ccdc_model.add(Flatten())
    ccdc_model.add(RepeatVector(2))
    ccdc_model.add(LSTM(32))
    ccdc_model.add(Dense(y_train.shape[1], activation='softmax'))
    #compile the model
    ccdc_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #train and load the model
    if os.path.exists("model/cnn_weights1.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights1.hdf5', verbose = 1, save_best_only = True)
        hist = ccdc_model.fit(X_train, y_train, batch_size = 32, epochs = 15, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/cnn_history1.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        ccdc_model.load_weights("model/cnn_weights1.hdf5")
    #perform prediction on test data  
    predict = ccdc_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    #call this function to calculate accuray and other metrics
    calculateMetrics("Propose Hybrid CCDC-HNN", y_test1, predict)

def values(filename, acc, loss):
    f = open(filename, 'rb')
    train_values = pickle.load(f)
    f.close()
    accuracy_value = train_values[acc]
    loss_value = train_values[loss]
    return accuracy_value, loss_value

def graph():
    train_acc, train_loss = values("model/cnn_history.pckl", "accuracy", "loss")
    val_acc, val_loss = values("model/cnn_history.pckl", "val_accuracy", "val_loss")

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy')
    plt.plot(train_acc, 'ro-', color = 'green')
    plt.plot(train_loss, 'ro-', color = 'blue')
    plt.plot(val_acc, 'ro-', color = 'red')
    plt.plot(val_loss, 'ro-', color = 'pink')
    plt.legend(['Training Accuracy', 'Training Loss', 'Validation Accuracy', 'Validation Loss'], loc='upper left')
    plt.title('CCDC-HNN Algorithm Training Accuracy & Loss Graph')
    plt.tight_layout()
    plt.show()

def predict():
    text.delete('1.0', END)
    global unet_model, ccdc_model
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename,0)
    image = img
    img = cv2.resize(img,(128, 128), interpolation = cv2.INTER_CUBIC)
    img = (img-127.0)/127.0
    img = img.reshape(1,128,128,1)
    preds = unet_model.predict(img)#predict segmented image
    preds = preds[0]
    cv2.imwrite("test.png", preds*255)
    img = cv2.imread(filename)
    img = cv2.resize(img,(128, 128), interpolation = cv2.INTER_CUBIC)
    mask = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128, 128), interpolation = cv2.INTER_CUBIC)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    output = "Benign"
    for bounding_box in bounding_boxes:
        (x, y, w, h) = bounding_box
        if w > 6 and h > 6:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            w = w + h
            output = "Malignant"
    img = cv2.resize(img, (300, 300))
    mask = preds*255
    mask = cv2.resize(mask, (300, 300))
    cv2.putText(img, output, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    cv2.imshow('Input Image', img)
    cv2.imshow('Cancer Detected Image', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()       

font = ('times', 16, 'bold')
title = Label(main, text='A Novel Hybrid Deep Learning Method for Early Detection of Lung Cancer using Neural Networks', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload LIDC-IDRI Dataset", command=uploadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Process Dataset", command=processDataset)
processButton.place(x=330,y=100)
processButton.config(font=font1) 

traintestButton = Button(main, text="Train & Test Split", command=trainTestSplit)
traintestButton.place(x=670,y=100)
traintestButton.config(font=font1) 

cnnButton = Button(main, text="Run Hybrid CCDC-HNN Algorithm", command=runCNN)
cnnButton.place(x=10,y=150)
cnnButton.config(font=font1)

predictButton = Button(main, text="Cancer Cell Detection & Classification", command=predict)
predictButton.place(x=330,y=150)
predictButton.config(font=font1)

graphButton = Button(main, text="CCDC-HNN Training Graph", command=graph)
graphButton.place(x=670,y=150)
graphButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=140)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)

main.config(bg='light coral')
main.mainloop()
