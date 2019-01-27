from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import os
import sys
start_time = time.time()
import pandas as pd
from operator import itemgetter
import skimage
import keras
from keras import models
from keras import layers
from keras import optimizers
import keras.callbacks
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import keras_preprocessing
from keras.callbacks import ModelCheckpoint
import numpy as np
from PIL import Image, ImageEnhance
from keras import optimizers
import os.path
from keras.models import load_model
from keras import applications
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
# from scipy.misc import imread
# from keras import models
from keras import backend as K
import math
import generic_utils
from image1 import ImageDataGenerator
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import keras.optimizers
import functools
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

'''

about the data :

Kindly provided by the Messidor program partners (see http://www.adcis.net/en/DownloadThirdParty/Messidor.html).
Users of the messidor database are encouraged to cite the following article:

Decencière et al.. Feedback on a publicly distributed database: the Messidor database.
Image Analysis & Stereology, v. 33, n. 3, p. 231-234, aug. 2014. ISSN 1854-5165.
Available at: http://www.ias-iss.org/ojs/IAS/article/view/1155 or
http://dx.doi.org/10.5566/ias.1155.

'''


df4 = pd.read_csv('diabetic_images.csv')
import pickle
df4.to_pickle("diabetic_images1.pickle")
df2 = pd.read_pickle("diabetic_images1.pickle")


df2 = df2.sort_values(by=['Image name'])
df2 = df2.reset_index(drop=True)



df2['classes'] = df2['Retinopathy grade'].mask(( df2['Retinopathy grade'] == 0 ) & (df2.iloc[:, 3] == 0), 0)
df2['classes'] = df2['Retinopathy grade'].mask(( df2['Retinopathy grade'] == 1 ) & (df2.iloc[:, 3] == 0), 1)
df2['classes'] = df2['Retinopathy grade'].mask(( df2['Retinopathy grade'] == 1 ) & (df2.iloc[:, 3] == 1), 1)
df2['classes'] = df2['Retinopathy grade'].mask(( df2['Retinopathy grade'] == 1 ) & (df2.iloc[:, 3] == 2), 1)
df2['classes'] = df2['Retinopathy grade'].mask(( df2['Retinopathy grade'] == 2 ) & (df2.iloc[:, 3] == 0), 1)
df2['classes'] = df2['Retinopathy grade'].mask(( df2['Retinopathy grade'] == 2 ) & (df2.iloc[:, 3] == 1), 1)
df2['classes'] = df2['Retinopathy grade'].mask(( df2['Retinopathy grade'] == 2 ) & (df2.iloc[:, 3] == 2), 1)
df2['classes'] = df2['Retinopathy grade'].mask(( df2['Retinopathy grade'] == 3 ) & (df2.iloc[:, 3] == 1), 2)
df2['classes'] = df2['Retinopathy grade'].mask(( df2['Retinopathy grade'] == 3 ) & (df2.iloc[:, 3] == 2), 2)


# After a few trials and errors in this data, I found out that sometimes less is more,
# In  order to create a model that will be able to clacify with higher accuracy,
# It needed to be feed with more  clearer  cases of
# diabetic retinopathy wich means reducing data even though the date was small, because the data was small!

# train set class_0
df2_0_train_set = df2[(df2['Retinopathy grade'] ==0) & (df2.iloc[: , 3] ==0)].head(100)

# test & validation set class_0
df2_0_test_set = df2[(df2['Retinopathy grade'] ==0) & (df2.iloc[:, 3] ==0)].tail(50)
# validation set class_0
df2_0_validation_set = df2_0_test_set.head(25)
# test set class_0
df2_0_test_set = df2_0_test_set.tail(25)

# train set class_2
df2_2_1_train_set = df2[(df2['Retinopathy grade'] ==3) & (df2.iloc[:, 3] ==1)].head(30)

df2_2_2_train_set = df2[(df2['Retinopathy grade'] ==3) & (df2.iloc[:, 3] ==2)].head(70)

df2_train_set_class_2 = df2_2_1_train_set.append([df2_2_2_train_set], ignore_index=False)


# test set class_2

df2_2_2_test_set = df2[(df2['Retinopathy grade'] == 3) & (df2.iloc[:, 3] == 2)].tail(38)

df2_2_2_test_set_1=df2_2_2_test_set.head(19)
df2_2_2_validation_set_1=df2_2_2_test_set.tail(19)
df2_2_1_test_set = df2[(df2['Retinopathy grade'] == 3) & (df2.iloc[:, 3] == 1)].tail(12)

df2_2_1_test_set_2=df2_2_1_test_set.tail(6)
df2_2_1_validation_set_2=df2_2_1_test_set.head(6)

df2_test_set_class_2 = df2_2_2_test_set_1.append([df2_2_1_test_set_2], ignore_index=False)
df2_validation_set_class_2=df2_2_2_validation_set_1.append([df2_2_1_validation_set_2], ignore_index=False)

# path= 'Base11_34'
#
# images = os.listdir('Base11_34')
#
# images= sorted(images)

# # building the CNN MODEL

# # creating the folders for train & test & validation
'''
# class_0 train_set
my_indices_0_train_set=df2_0_train_set.index.values.tolist()
images_0_train_set = itemgetter(*my_indices_0_train_set)(images)
for image in images_0_train_set :
 img_0 = Image.open('D:/ליאת/image_diabetes/Base11_34' + '/' + image)
 img_0 = img_0.resize((150, 150), Image.ANTIALIAS)
 img_0.save('train_set/class_0' + '/' + image)

 # class_0 test_set
my_indices_0_test_set=df2_0_test_set.index.values.tolist()
images_0_test_set = itemgetter(*my_indices_0_test_set)(images)
for image in images_0_test_set :
 img_0 = Image.open('D:/ליאת/image_diabetes/Base11_34' + '/' + image)
 img_0 =img_0.resize((150, 150), Image.ANTIALIAS)
 img_0.save('test_set/class_0' + '/' + image)

 # class_0 validation_set
my_indices_0_validation_set=df2_0_validation_set.index.values.tolist()
images_0_validation_set = itemgetter(*my_indices_0_validation_set)(images)
for image in images_0_validation_set :
 img_0 = Image.open('D:/ליאת/image_diabetes/Base11_34' + '/' + image)
 img_0 =img_0.resize((150, 150), Image.ANTIALIAS)
 img_0.save('validation_set/class_0' + '/' + image)

# class_2 train_set
my_indices_2_train_set=df2_train_set_class_2.index.values.tolist()
images_2_train_set = itemgetter(*my_indices_2_train_set)(images)
for image in images_2_train_set :
 img_2 = Image.open('D:/ליאת/image_diabetes/Base11_34' + '/' + image)
 img_2 =img_2.resize((150, 150), Image.ANTIALIAS)
 img_2.save('train_set/class_2' + '/' + image)

# class_2 test_set
my_indices_2_test_set = df2_test_set_class_2.index.values.tolist()
images_2_test_set = itemgetter(*my_indices_2_test_set)(images)
for image in images_2_test_set:
 img_2 = Image.open('D:/ליאת/image_diabetes/Base11_34' + '/' + image)
 img_2 =img_2.resize((150, 150), Image.ANTIALIAS)
 img_2.save('test_set/class_2' + '/' + image)

 # class_2 validation_set
my_indices_2_validation_set=df2_validation_set_class_2.index.values.tolist()
images_2_validation_set = itemgetter(*my_indices_2_validation_set)(images)
for image in images_2_validation_set :
 img_2 = Image.open('D:/ליאת/image_diabetes/Base11_34' + '/' + image)
 img_2 =img_2.resize((150, 150), Image.ANTIALIAS)
 img_2.save('validation_set/class_2' + '/' + image)

'''

nb_train_samples = 200
nb_validation_samples = 50
epochs =10
batch_size = 40

# model = load_model('my_model45.h5') ## instead of running the model, there is saved one ## 0.90 - 0.92
# using pretrained model, because of the limited data (small)

base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
# Freeze the layers except the last 4 layers
# for layer in base_model.layers[:-4]:
for layer in base_model.layers[:-4]:
    layer.trainable = False
model_top = Sequential()
model_top.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:], data_format=None)),
model_top.add(Dense(4, activation='relu'))
model_top.add(Dropout(0.5))
model_top.add(Dense(1, activation='sigmoid'))
model = Model(inputs=base_model.input, outputs=model_top(base_model.output))
# load weights
import pickle
# saving weights as a pickle:

# with open("weights.best.180.pickle", 'wb') as f:
#     pickle.dump("weights.best.hdf180", f)

#loading weights :
with open("weights.best.180.pickle", 'rb') as f:
 weights_best_hdf180 = pickle.load(f)

model.load_weights(weights_best_hdf180)
model.compile(keras.optimizers.Adam(lr=0.00), loss='binary_crossentropy', metrics=['accuracy'])


# # adding augmentation to the documentation (marked with #, increased the accuracy from 86-88 to 90-94 )
train_datagen = ImageDataGenerator(

        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        histogram_equalization= True,##
        contrast_stretching=True,##
        adaptive_equalization=True,##
        contrast=True,##
        sharpness=True,##
        adaptive_extraction=True,##
        invert=True,##
        rotation_range=90,
        vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
      'train_set',

      target_size=(150, 150),
      batch_size=batch_size,
      class_mode='binary', shuffle=True)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(

        'test_set',
        target_size=(150, 150),
        batch_size=1,
        class_mode='binary', shuffle=True)

validation_generator = test_datagen.flow_from_directory(

        'validation_set',
        target_size=(150, 150),
        batch_size=1,
        class_mode='binary', shuffle=False)

filepath = "weights.best.hdf184"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


class LRFinder(Callback):
        def __init__(self, min_lr=0.00001, max_lr=0.009, steps_per_epoch=None, epochs=None):
            super().__init__()

            self.min_lr = min_lr
            self.max_lr = max_lr
            self.total_iterations = steps_per_epoch * epochs
            self.iteration = 0
            self.history = {}

        def clr(self):
            '''Calculate the learning rate.'''
            x = self.iteration / self.total_iterations
            return self.min_lr + (self.max_lr - self.min_lr) * x

        def on_train_begin(self, logs=None):
            '''Initialize the learning rate to the minimum value at the start of training.'''
            logs = logs or {}
            K.set_value(self.model.optimizer.lr, self.min_lr)

        def on_batch_end(self, epoch, logs=None):
            # Record previous batch statistics and update the learning rate.
            logs = logs or {}
            self.iteration += 1

            self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
            self.history.setdefault('iterations', []).append(self.iteration)

            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

            K.set_value(self.model.optimizer.lr, self.clr())

        def plot_lr(self):
            # Helper function to quickly inspect the learning rate schedule.
            plt.plot(self.history['iterations'], self.history['lr'])
            plt.yscale('log')
            plt.xlabel('Iteration')
            plt.ylabel('Learning rate')
            plt.show()
            #

        def plot_loss(self):
            # '''Helper function to quickly observe the learning rate experiment results.'''
            plt.plot(self.history['lr'], self.history['loss'])
            plt.xscale('log')
            plt.xlabel('Learning rate')
            plt.ylabel('Loss')
            plt.show()


# monitoring the best range/ lrate
lr_finder = LRFinder(min_lr=0.00001,max_lr=0.004, steps_per_epoch=np.ceil(nb_train_samples // batch_size), epochs=10)

# callbacks_list = [checkpoint]
callbacks_list = [checkpoint, lr_finder]

model.fit_generator(
       train_generator,
       steps_per_epoch=nb_train_samples // train_generator.batch_size,
       epochs=epochs,
       validation_data=test_generator,
       validation_steps=nb_validation_samples // test_generator.batch_size,
       verbose=1, callbacks=callbacks_list,
       workers=16)

lr_finder.plot_loss()





nb_samples = len(validation_generator.filenames)

pred = model.predict_generator(validation_generator, steps=nb_samples)

fpr, tpr, thresholds_keras = roc_curve(validation_generator.classes, pred)



auc = auc(fpr, tpr)
print("AUC : " , auc)  ## 0.92

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr,tpr, label='Keras (area = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()

# 42 0.00001-005
# 44 0.00001-009

# saving the model
# model.save('my_model48.h5')

print("--- %s seconds ---" % (time.time() - start_time))





