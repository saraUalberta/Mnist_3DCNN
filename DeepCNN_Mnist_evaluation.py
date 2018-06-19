# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:52:57 2017
PPMI Image Classification using Stack of Convolutional Auto Encoder
@author: sara
"""
## Keras
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D,UpSampling3D,Input)
from keras.models import Model
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.optimizers import Adadelta
## others
import numpy as np
import os
FLOAT_PRECISION = np.float32
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import ipdb
from sklearn.neighbors import KNeighborsClassifier

## subfunction
def load_newdataset():
    with h5py.File("/home/sara/Documents/Sara_Thesis/lastDataSet/train_point_clouds.h5", "r") as hf:    
    	X_train = hf["X_train"][:]
    	y_train = hf["y_train"][:]    
     	X_test = hf["X_test"][:]  
     	y_test = hf["y_test"][:]  
def convert_to_binary(Y_test):
    y_pred = np.zeros([Y_test.shape[0],1],dtype=np.int)
    for i in range(len(Y_test)):
	if Y_test[i,0]==1.0:
            y_pred[i] = 1
        elif Y_test[i,1]==1.0:
            y_pred[i] = 0
    return(y_pred)
def plot_history(history, result_dir):
    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'Mnis_model_loss.png'))
    plt.close()
def extract_feature(autoencoders, x,bs,ln):
    hidden_layer = autoencoders.layers[ln]
    feature_model = Model(autoencoders.input, hidden_layer.output)
    return feature_model.predict(x, batch_size=bs)
def visualize_res(imgo,imgrec,fn,orglabeltr):
    frame = 15
    simg1 = imgo[100,:,:,frame]   
    simg2 = imgo[200,:,:,frame]
    rimg1 = imgrec[100,:,:,frame]   
    rimg2 = imgrec[200,:,:,frame]
    print(orglabeltr[100],orglabeltr[200])
    plt.subplot(2, 2, 1)
    plt.imshow(simg1)
    plt.title('sample image 1 with label %d'%orglabeltr[100])
    plt.subplot(2, 2, 2)
    plt.imshow(simg2)
    plt.title('sample image 2with label %d'%orglabeltr[200])
    plt.subplot(2, 2, 3)
    plt.imshow(rimg1)
    plt.title('Recons sample image 1')
    plt.subplot(2, 2, 4)
    plt.imshow(rimg2)
    plt.title('Recons sample image 2')
    plt.savefig(fn)
    plt.show()
def check_datadistribution():
    print('Data Distribution')
    
def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig('model_accuracy.png')
    plt.close()
    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig('model_loss.png')
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc'] 
    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']
    acc = np.array(acc)
    val_acc = np.array(val_acc)
    Training_Acc = acc.mean()
    Validation_Acc = val_acc.mean()
    nb_epoch = len(acc)

    with open('result.txt', 'w') as fp:
        fp.write('epoch\tacc\tval_acc\tloss\tval_loss\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
i, acc[i],val_acc[i],loss[i], val_loss[i]))
    return (Training_Acc, Validation_Acc)
## Parameters
channel = 1
nb_classes = 10
epoch = 200
batch = 128
output = 'Mnist_3DCAE'
layernum = 1
channel =1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# create 3D mnist dataset
print(x_train.shape,x_test.shape)
img_dim = 30
frames = img_dim
Mnist3D_train = np.zeros((x_train.shape[0],img_rows, img_cols,img_dim),dtype =np.float32)
for i in range(x_train.shape[0]):
    cur_img = x_train[i]
    for j in range(img_dim-1):
        Mnist3D_train[i,:,:,j] = cur_img
print(Mnist3D_train.shape)
 
Mnist3D_test = np.zeros((x_test.shape[0],img_rows, img_cols,img_dim),dtype =np.float32)
for i in range(x_test.shape[0]):
    cur_img = x_test[i]
    for j in range(img_dim-1):
        Mnist3D_test[i,:,:,j] = cur_img
print(Mnist3D_test.shape)


X_train = Mnist3D_train.reshape((Mnist3D_train.shape[0], img_rows, img_cols, frames, channel))
X_test = Mnist3D_test.reshape((Mnist3D_test.shape[0], img_rows, img_cols, frames, channel))
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print(X_train.shape, X_test.shape)
print(Y_train.shape,Y_test.shape)
'''
input_img = Input(shape=(28, 28, img_dim,1))

## Model Structure
x = Conv3D(8, (3, 3,3), activation='relu', padding='same')(input_img)
x = MaxPooling3D((2, 2,1), padding='same')(x)

x = Conv3D(8, (3, 3,3), activation='relu', padding='same')(x)
#x = MaxPooling3D((2, 2,1), padding='same')(x)

#x = Conv3D(8, (3, 3,3), activation='relu', padding='same')(x)
#x = MaxPooling3D((2, 2,1), padding='same')(x)

x = Conv3D(8, (3, 3,3), activation='relu', padding='same')(x)
encoded = MaxPooling3D((2, 2,1), padding='same', name='encoder')(x)


x = Conv3D(8, (3, 3,3), activation='relu', padding='same')(encoded)
x = UpSampling3D(size=(2, 2, 1))(x)

x = Conv3D(8, (3, 3,3), activation='relu', padding='same')(x)
#x = UpSampling3D(size=(2, 2, 1))(x)

x = Conv3D(8, (3, 3,3), activation='relu', padding='same')(x)
x = UpSampling3D(size=(2, 2, 1))(x)


#x = Conv3D(8, (3, 3,3), activation='relu', padding='same')(x)
#x =UpSampling3D(size=(2, 2, 1))(x)

decoded = Conv3D(1, (3, 3,3), activation='relu', padding='same',name='decoder')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()
plot_model(autoencoder, show_shapes=True,to_file=os.path.join(output, 'Mnis_CAE.png'))

#autoencoder.compile(loss='mean_squared_error', optimizer='rmsprop')

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
autoencoder.compile(optimizer='Adadelta', loss='mse')
#autoencoder.compile(loss='mean_squared_error', optimizer=sgd,metrics=['binary_accuracy'])
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
history = autoencoder.fit(X_train, X_train, validation_data=(X_test, X_test), batch_size=batch,
                        epochs=epoch, verbose=1, shuffle=True)
## saving the model
model_json = autoencoder.to_json()    
if not os.path.isdir(output):
    os.makedirs(output)
with open(os.path.join(output, 'Mnist_CAE.json'), 'w') as json_file:
    json_file.write(model_json)
autoencoder.save_weights(os.path.join(output, 'Mnis_CAE.hd5'))
#plot_history(history, output)

#np.save('Mnis_history.npy', history) 

ipdb.set_trace()
'''
## classification

minstmod = 'Mnist_CAE.json'
mnistwna = 'Mnist_CAE.hd5'

json_file = open(minstmod, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(mnistwna)
print("Loaded model from disk")
## visualize the decoded image
decoded_imgs = loaded_model.predict(X_test,batch_size=batch)
decoded_img_train = loaded_model.predict(X_train,batch_size=batch)
## 

# Test
X_test_r = np.reshape(X_test,(10000,28,28,img_dim))# its shape has 1 at the end and wana get rid of that
decoded_imgs_te = np.reshape(decoded_imgs,(10000,28,28,img_dim))
fn = 'TestImg'
visualize_res(X_test_r,decoded_imgs_te,fn,y_test)
# Train
X_train_r = np.reshape(X_train,(60000,28,28,img_dim))
decoded_imgs_tr = np.reshape(decoded_img_train,(60000,28,28,img_dim))
## print labels
fn = 'TrainImg'
visualize_res(X_train_r,decoded_imgs_tr,fn,y_train)
## Visualize Encoder output feature maps

ln = 5
frame = 15
fmapa =8
sn1 = 100
sn2 = 200
enc_output = extract_feature(loaded_model,X_train,batch,ln)
print(enc_output.shape)
print('Label for sample image is:''s1 label',y_train[sn1], 's2 label',y_train[sn2])
fig=plt.figure()
for i in range(fmapa):
    print(i+1)
    cur_fmap = enc_output[:,:,:,:,i]
    simg1 = cur_fmap[sn1,:,:,frame]   
    simg2 = cur_fmap[sn2,:,:,frame]  
    plt.subplot(2, fmapa, (i)+1)
    plt.imshow(simg1)
    plt.title('fmap %d'% i)
    plt.subplot(2, fmapa, (i)+9)
    plt.imshow(simg2)
    plt.title('fmap %d' % i)
    print(i+2)
plt.suptitle('S1 label and S2 Label is:\n'+str(y_train[sn2])+'and'+str(y_train[sn1]))
plt.savefig('featuremapnorm.png')
plt.show()

loaded_model.compile(optimizer='Adam', loss='mse')
print("3DCAE Scoring:")
scoring = loaded_model.evaluate(X_test, X_test,batch_size=2, verbose=1)
print(loaded_model.metrics_names)
print(scoring)
##


Y_tr  = y_train#np.ravel(convert_to_binary(y_train))
Y_te  = y_test#np.ravel(convert_to_binary(y_test))
fsize_encoder = 7*7*img_dim*8

orgfeatures_tro = extract_feature(loaded_model,X_train,batch,5)
print(orgfeatures_tro.shape)
orgfeatures_tr = np.reshape(orgfeatures_tro,(orgfeatures_tro.shape[0], fsize_encoder))
orgfeatures_teo= extract_feature(loaded_model,X_test,batch,5)
print(orgfeatures_teo.shape)
orgfeatures_te = np.reshape(orgfeatures_teo,(orgfeatures_teo.shape[0], fsize_encoder))
X_tr = orgfeatures_tr
X_te = orgfeatures_te
print(orgfeatures_tr.shape,orgfeatures_te.shape)

print('##################  Softmax Classification############################')
Gen_Acc=[]
model = Sequential()
model.add(Flatten(input_shape=orgfeatures_tro.shape[1:]))
model.add(BatchNormalization())
model.add(Dropout(0.5))
# first FC
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
# second FC
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
# third FC
#model.add(Dense(20, activation='relu'))
#model.add(Dropout(0.5))
# forth FC
#model.add(Dense(20, activation='relu'))
#model.add(Dropout(0.5))
# output layer
model.add(Dense(nb_classes, activation='softmax'))
##  initiate RMSprop optimizer
print('before fitting')
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-06, decay=0.0)
model.compile(loss='categorical_crossentropy',optimizer= Adam(),metrics=['accuracy'])
history = model.fit(orgfeatures_tro, Y_train, validation_data=(orgfeatures_teo, Y_test), batch_size=batch,
                        epochs=epoch, verbose=1, shuffle=True)

loss, acc = model.evaluate(orgfeatures_teo, Y_test, verbose=0,batch_size=batch) 
Tr_Acc , Val_Acc = save_history(history, output)                      
print('Test accuracy:', acc)
losstr, acctr =  model.evaluate(orgfeatures_tro, Y_train, verbose=0,batch_size=batch)
Gen_Acc.append(acctr)
Gen_Acc.append(acc)
y_pred =  model.predict(orgfeatures_teo,batch_size=batch)

y_pred2 = np.ravel(convert_to_binary(y_pred))
Gen_YPred = y_pred
print(Gen_YPred)
print(Gen_Acc)
np.savetxt('ypred_encoutput_agesex.txt', Gen_YPred)
np.savetxt('Acc_encoutput_agesex.txt', Gen_Acc)
plot_history(history, output)

'''
print('################## 2-KNN Classification############################')

knn = KNeighborsClassifier()
knn.fit(orgfeatures_tr,y_train)
acc = knn.score(orgfeatures_te, y_test)

print("Training accuracy: {:.2f}%".format(knn.score(X_train, y_train)))
print("accuracy: {:.2f}%".format(acc ))
print('y true', y_test)
print('y pred', knn.predict(X_train))

'''
