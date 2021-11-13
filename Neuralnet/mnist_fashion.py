import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
# we are dealing with images of 28*28 pixels, which will be flatten to 784 input for our NN
# 128 hidden neurons (1 layer)
# 10 output neurons
# activation function = relu

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
class_name = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneakers', 'Bag', 'Ankle boot']

train_images = train_images/255
test_images = test_images/255

model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)), keras.layers.Dense(128, activation="relu"),
                          keras.layers.Dense(10, activation="softmax")])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)

"""
test_loss, test_acc = model.evaluate(test_images, test_labels)
print ("Tested Accuracy: ", test_acc)
"""

# test the model by predicting what test image 7 is
prediction = model.predict(test_images)
for i in range(5):
    pic = test_images[i]
    pic = cv2.resize(pic,(600,600))
    info = "actual: " + class_name[test_labels[i]] + "  " + "prediction: " + class_name[np.argmax(prediction[i])]
    pic = cv2.putText(pic,info,(10,10),cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
    cv2.imshow("info", pic)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
