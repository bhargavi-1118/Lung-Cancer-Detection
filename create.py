import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv3D, MaxPooling3D, LSTM,RepeatVector
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import pickle
import os

X = []
Y = []

for root, dirs, directory in os.walk("trainingImages"):
    for j in range(len(directory)):
        name = os.path.basename(root)
        img = cv2.imread(root+"/"+directory[j])
        img = cv2.resize(img,(32, 32), interpolation = cv2.INTER_CUBIC)
        img = img.astype('float32')
        img = img/255
        X.append([img])
        if name == 'Normal':
            Y.append(0)
        else:
            Y.append(1)
        print(name)    
X = np.asarray(X)
Y = np.asarray(Y)
np.save("model/X2",X)
np.save("model/Y2",Y)

X = np.load("model/X2.npy")
Y = np.load("model/Y2.npy")

print(X.shape)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test

cnn_model = Sequential()
cnn_model.add(Conv3D(32, (1, 3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], X.shape[3], X.shape[4])))
cnn_model.add(MaxPooling3D((1, 2, 2)))
cnn_model.add(Conv3D(16, (1, 3, 3), activation='relu'))
cnn_model.add(MaxPooling3D((1, 2, 2)))
cnn_model.add(Conv3D(16, (1, 3, 3), activation='relu'))
cnn_model.add(MaxPooling3D((1, 2, 2)))
cnn_model.add(Flatten())
cnn_model.add(RepeatVector(2))
cnn_model.add(LSTM(32))
cnn_model.add(Dense(y_train.shape[1], activation='softmax'))
cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
if os.path.exists("model/cnn_weights1.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights1.hdf5', verbose = 1, save_best_only = True)
    hist = cnn_model.fit(X_train, y_train, batch_size = 32, epochs = 35, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history1.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn_model.load_weights("model/cnn_weights1.hdf5")
   
predict = cnn_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test1, predict)
print(acc)





