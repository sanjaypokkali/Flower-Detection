
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

df=pd.read_csv('C:/Users/sanja/Desktop/flowerdet/data/train.csv')

df2=pd.read_csv('C:/Users/sanja/Desktop/flowerdet/data/test.csv')

y=df['category'].values


images=[]

type(images)

from PIL import Image
import numpy as np
def load_images(folder,start):
  if start!=18000:
    for i in range(start,start+1000):
      if 'DS_Store' not in "hello":
        image = Image.open(os.path.join(folder,str(i)+".jpg"))
        if image is not None:
          h = image.size[0]
          w = image.size[1]
          image = np.array(image.resize((500,500)))
          image= image / 255.
          images.append(image)
  else:
    for i in range(start,start+540):
      image = Image.open(os.path.join(folder,str(i)+".jpg"))
      if image is not None:
        h = image.size[0]
        w = image.size[1]
        image = np.array(image.resize((500,500)))
        image= image / 255.
        images.append(image)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (500,500,3)))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(103, activation = "softmax"))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


y=to_categorical(y)


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


images=[]
for i in range(19):
  images.clear()
  load_images('C:/Users/sanja/Desktop/flowerdet/data/train',i*1000)
  X=np.array(images)
  if i!=18:
    y_train=y[i*1000:i*1000+1000]
  else:
    y_train=y[i*1000:i*1000+540]
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
  datagen.fit(X_train)
  model.fit_generator(datagen.flow(X_train,y_train, batch_size=1000),
                              epochs = 15, validation_data = (X_test,y_test),
                              verbose = 1, steps_per_epoch=X_train.shape[0]/batch_size)
  print(str(i)+"done")

images_test=[]

def load_images_test(folder):
  for i in range(18540,20549):
      if 'DS_Store' not in "hello":
        image = Image.open(os.path.join(folder,str(i)+".jpg"))
        if image is not None:
          h = image.size[0]
          w = image.size[1]
          image = np.array(image.resize((500,500)))
          image= image / 255.
          images_test.append(image)

len(images_test)
load_images_test('C:/Users/sanja/Desktop/flowerdet/data/test')
X_test=np.array(images_test)
pred=model.predict_classes(X_test)

len(pred)

df2['category']=pred

df2.to_excel('submission1.xlsx', index=False)

df2.to_excel('submission1.xlsx', index=False)
