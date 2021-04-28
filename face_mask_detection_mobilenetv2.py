
"""
@author : Umang Agarwal

"""

from google.colab import drive
drive.mount('/content/drive')

#IMPORTING LIBRARIES

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG19
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2

# PREPROCESSING
" LOAD ALL IMAGES AND LABELS AND PROCESS THEM , CONVERT TO NUMPY ARRAYS AND APPEND THEM TO THE REPECTIVE LISTS AND CONVERTING THOSE TO NUMPY ARRAYS"

path = '/content/drive/My Drive/maskclassifier/train/'
imagePaths = list(paths.list_images(path))
images = []
labels = []
for imagePath in imagePaths:
	label = imagePath.split(os.path.sep)[-2]
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)
	images.append(image)
	labels.append(label)
images = np.array(images, dtype="float32")
labels = np.array(labels)

print(images.shape)  # SHAPE OF IMAGES
print(labels.shape)  # SHAPE OF LABELS

print(np.unique(labels)) # VIEWING THE LABELS

# ONE HOT ENCODE THE LABELS AS THEY ARE CATEGORICAL

encoder = LabelBinarizer()
labels = encoder.fit_transform(labels)
labels = to_categorical(labels)

# PERFROM THE TRAIN TEST SPLIT BY FORMING GIVING 20% DATASET TO TEST OUR MODEL 

X_train, X_test, y_train, y_test = train_test_split(images, labels,
	test_size=0.20, stratify=labels)

# TRAINING IMAGE GENERATOR FOR DATA AUGMENTATION

datagen = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")


#MOBILENETV2 MODEL 
" FOR THIS TASK WE WILL BE FINE TUNING THE MOBILENETV2 ARCHITECTURE, A HIGHLY EFFECIENT ARCHITECTURE WHICH WORKS WELL WITH LIMITED COMPUTATIONAL CAPACITY "
"KERAS FUNCTIONAL API HAS BEEN USED TO MAKE THE ARCHITECTURE OF THE MODEL "

baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
X = baseModel.output
X = AveragePooling2D(pool_size=(7, 7))(X)
X = Flatten()(X)
X = Dense(128, activation="relu")(X)
X = Dropout(0.5)(X)
X = Dense(2, activation="softmax")(X)
model = Model(inputs=baseModel.input, outputs=X)

" AS WE ARE USING TRANSFER LEARNING IE PRETRAINED MOBILENETV2 WE NEED TO FREEZE ITS LAYERS AND TRAIN ONLY LAST TWO DENSE LAYERS"

for layer in baseModel.layers:
	layer.trainable = False

model.summary()

# DEFINING FEW PARAMETERS
batch_size = 128
epochs = 15

# DEFINING THE OPTIMIZER AND COMPILING THE MODEL
optimizer = Adam(lr=1e-4, decay=1e-3)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

#TRAINING THE MODEL

hist = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), 
            steps_per_epoch=len(X_train) // batch_size,
			validation_data=(X_test, y_test),
			validation_steps=len(X_test) // batch_size,
			epochs=epochs)

"WE NEED TO FIND THE INDEX OF THE LABEL WITH CORRESPONDING LARGEST PREDICTED PROBABILITY FOR EACH IMAGE IN TEST SET"

y_pred = model.predict(X_test, batch_size=batch_size)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(y_test.argmax(axis=1), y_pred, target_names=encoder.classes_))

#SAVING THE MODEL.H5 FILE SO THAT IT CAN BE LOADED LATER TO USE FOR MASK DETECTION
model.save("model", save_format="h5")

# PLOT THE TRAIN AND VALIDATION LOSS FOR OUR MODEL USING MATPLOTLIB LIBRARY
plt.plot(np.arange(0, epochs), hist.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), hist.history["val_loss"], label="val_loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="upper right")

# WE HAVE USED PRETRAINED MODEL TO DETECT FACES IN IMAGES AND USED OPENCV DEEP NEURAL NETWORK MODULE TO READ MODEL AND CONFIG FILE
# WEIGHTS OF THE TRAINED MASK CLASSIFIER MODEL IN LOADED

prototxtPath = '/content/drive/My Drive/maskclassifier/face_detector/deploy.prototxt'
weightsPath = '/content/drive/My Drive/maskclassifier/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
face_model = cv2.dnn.readNet(prototxtPath, weightsPath)
model = load_model("model")

"""PREPROCESS THE IMAGES USING BLOB MODULE OF OPENCV WHICH RIESIZES AND CROPS IMAGES FROM CENTER, SUBSTRACT MEANS VALUES,
SCALES VALUES BY SCALEFACTOR, SWAP BLUES AND RED CHANNELS AND THEN PASS THE BLOB THROUGH
OUR NETWORK TO OBTAIN THE FACE WHICH ARE DETECTED BY THE MODEL """


mage = cv2.imread('/content/drive/My Drive/maskclassifier/test/people1.png')
# image = cv2.imread('/content/drive/My Drive/maskclassifier/test/people2.jpg')
height, width = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
face_model.setInput(blob)
detections = face_model.forward()			#detecting the faces

""" IN THIS PART WE HAVE LOOP THROUGH ALL THE DETECTIONS AND IF THEIR SCORE IS GREATER THAN
CERTAIN THRESHOLD THEN WE HAVE FIND THE DIMENSIONS FOR FACE AND USE PREPROCESSING
DTEPS USE FOR TRAINING IMAGES. THEN WE HAVE USED MODEL TRAINED TO PREDICT THE CLASS OF FACE IMAGE
BY PASSING THE IMAGE THROUGH IT. """

""" THE OPENCV FUNCTIONS ARE USED TO CREATE BOUNDING BOXES, PUT TEXT AND SHOW THE IMAGE"""

from google.colab.patches import cv2_imshow
threshold = 0.2
person_with_mask = 0;
person_without_mask = 0;
for i in range(0, detections.shape[2]):
	score = detections[0, 0, i, 2]
	if score > threshold:
		#coordinates of the bounding box
		box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
		X_start, Y_start, X_end, Y_end = box.astype("int")
		X_start, Y_start = (max(0, X_start), max(0, Y_start))
		X_end, Y_end = (min(width - 1, X_end), min(height - 1, Y_end))

		face = image[Y_start:Y_end, X_start:X_end]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)		#Convert to rgb
		face = cv2.resize(face, (224, 224))					#resize
		face = img_to_array(face)							
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)	
	
		mask, withoutMask = model.predict(face)[0]			#To predict mask or not on the face

		if mask > withoutMask:								#determining the label
			label = "Mask"
			person_with_mask += 1
		else: 
			label = "No Mask"
			person_without_mask += 1
			
		if label == "Mask":									#determine the color
			color = (0, 255, 0)
		else:
			color = (0, 0, 255)
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)		#label and probability
		cv2.putText(image, label, (X_start, Y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (X_start, Y_start), (X_end, Y_end), color, 2)
  
print("Number of person with mask : {}".format(person_with_mask))
print("Number of person without mask : {}".format(person_without_mask))
cv2_imshow(image)


    
