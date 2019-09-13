
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

df=pd.read_csv('C:/Users/sanja/Desktop/flowerdet/data/train.csv')

df2=pd.read_csv('C:/Users/sanja/Desktop/flowerdet/data/test.csv')

y=df['category'].values
print(y.shape[0])

images=[]

type(images)

from PIL import Image
import numpy as np
def load_images(folder):
    for i in range(y.shape[0]):
      if 'DS_Store' not in "hello":
        image = Image.open(os.path.join(folder,str(i)+".jpg"))
        if image is not None:
          h = image.size[0]
          w = image.size[1]
          image = np.array(image.resize((150,150)))
          image= image / 255.
          images.append(image)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image

model = Sequential()
model.add(Conv2D(512, kernel_size=(3, 3),activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(103, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


y=to_categorical(y)
load_images('C:/Users/sanja/Desktop/flowerdet/data/train')
X=np.array(images)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
model.fit(X_train,y_train,epochs=50,validation_data=(X_test,y_test))

images_test=[]

def load_images_test(folder):
  for i in range(18540,20549):
      if 'DS_Store' not in "hello":
        image = Image.open(os.path.join(folder,str(i)+".jpg"))
        if image is not None:
          h = image.size[0]
          w = image.size[1]
          image = np.array(image.resize((150,150)))
          image= image / 255.
          images_test.append(image)

len(images_test)
load_images_test('C:/Users/sanja/Desktop/flowerdet/data/test')
X_test=np.array(images_test)
pred=model.predict_classes(X_test)

len(pred)
df2['category']=pred
df2.to_excel('submission1.xlsx', index=False)
