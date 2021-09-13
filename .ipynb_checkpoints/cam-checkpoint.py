import numpy as np
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
import cv2 as cv
import os
import pickle
import time
import random
import matplotlib.pyplot as plt 


#load model
model = keras.models.load_model("model.h5")
img_size = 70
categories= ["dog", "cat"]

def prepare(f):
    pic = cv.resize(f, (img_size, img_size))
    pic = pic.reshape(-1, img_size, img_size,3)
    return pic/255.0


from keras.preprocessing.image import ImageDataGenerator,load_img
def classify(image):
    frame = cv.imread(image)
    frame = prepare(frame)
    predictions = model.predict([frame])
    answer = categories[round(predictions[0][0])]
    print(answer.upper())
    return answer

    
font                   = cv.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2



cam = cv.VideoCapture(0)


while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break 
    cv.imshow("image", frame)
    

    k = cv.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "test.jpg"
        cv.imwrite(img_name, frame)
        print("{} written!".format(img_name))

cam.release()
cv .destroyAllWindows()

image = "test.jpg"
answer = classify(image)

