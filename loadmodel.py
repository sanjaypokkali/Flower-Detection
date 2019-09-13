import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image
import os
import numpy as np
import pandas as pd

model=load_model('test.h5')
df2=pd.read_csv('C:/Users/sanja/Desktop/flowerdet/data/test.csv')
images_test=[]

def load_images_test(folder):
  for i in range(18540,20549):
      if 'DS_Store' not in "hello":
        image = Image.open(os.path.join(folder,str(i)+".jpg"))
        if image is not None:
          h = image.size[0]
          w = image.size[1]
          image = np.array(image.resize((100,100)))
          image= image / 255.
          images_test.append(image)

len(images_test)
load_images_test('C:/Users/sanja/Desktop/flowerdet/data/test')
X_test=np.array(images_test)
pred=model.predict_classes(X_test)

len(pred)
df2['category']=pred
df2.to_excel('submission1.xlsx', index=False)
