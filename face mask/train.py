from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
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
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

init_lr=1e-4 #initial learning rate
epochs=5
bs=32
dir=r"F:\face mask detection\dataset"
categories=["with_mask","without_mask"]
print("*************loading images*************")
data=[]
labels=[]

for category in categories:
    path=os.path.join(dir,category)
    for img in os.listdir(path):
        imgpath=os.path.join(path,img)
        image=load_img(imgpath,target_size=(224,224))
        image=img_to_array(image)
        image=preprocess_input(image)
        data.append(image)
        labels.append(category)


#one hot ie with mask=1 without=0
lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

data=np.array(data,dtype="float32")
labels=np.array(labels)
(trainx, testx, trainy, testy)=train_test_split(data,labels,test_size=0.20,stratify=labels,random_state=42) #20% is val 80% is test

#image ImageDataGenerator creates many images with a single image with respect to flips, rotations etc
aug=ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest" )

basemodel=MobileNetV2(weights="imagenet",include_top=False, input_tensor=Input(shape=(224,224,3)))

headmodel = basemodel.output
headmodel = AveragePooling2D(pool_size=(7, 7))(headmodel)
headmodel = Flatten(name="flatten")(headmodel)
headmodel = Dense(128, activation="relu")(headmodel)
headmodel = Dropout(0.5)(headmodel)
headmodel = Dense(2, activation="softmax")(headmodel)

model=Model(inputs=basemodel.inputs,outputs=headmodel)

for layer in basemodel.layers:
    layer.trainable=False


# compile our model
print("***********compiling model*************")
opt = Adam(lr=init_lr, decay=init_lr / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network
print("**********training head************")
H = model.fit(
	aug.flow(trainx, trainy, batch_size=bs),
	steps_per_epoch=len(trainx) // bs,
	validation_data=(testx, testy),
	validation_steps=len(testx) // bs,
	epochs=epochs)

# make predictions on the testing set
print("*************** evaluating network**************")
predIdxs = model.predict(testx, batch_size=bs)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testy.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serialize the model to disk
print("*************saving mask detector model***************")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
'''N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")'''
