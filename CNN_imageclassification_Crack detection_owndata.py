# -*- coding: utf-8 -*-
"""
Created on Tue May  2 00:12:21 2023

@author: Rakshya Shrestha
"""

#importing libraries
import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import roc_curve, auc


#setting working directory
crack_dir = Path('C:/Users/14098/OneDrive - Lamar University/Desktop/Image/Cracks')
nocrack_dir = Path('C:/Users/14098/OneDrive - Lamar University/Desktop/Image/Nocracks')

#defining the function that takes a directory containing image files 
#and their respective labels, 
#and converts them into a Pandas DataFrame object 
def preprocessing_df(image_dir, label):
    filepaths = pd.Series(list(image_dir.glob(r'*.jpg')), name='Filepath').astype(str)
    labels = pd.Series(label, name='Label', index=filepaths.index)
    df = pd.concat([filepaths, labels], axis=1)
    return df

#creating a dataframe
crack_df = preprocessing_df(crack_dir, label="crack")
nocrack_df = preprocessing_df(nocrack_dir, label="nocrack")

#concatinating crack and uncrack images
both_df = pd.concat([crack_df, nocrack_df], axis=0).sample(frac=1.0, random_state=3).reset_index(drop=True)
both_df

#splitting the data into training and testing(80-20)
train_df, test_df = train_test_split(
    both_df,
    train_size=0.8,
    shuffle=True,
    random_state=7)

#scaling down the pixel values of each image by a factor of 1/255 to normalizes the image data between 0 and 1 for train data.
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255#,
    #validation_split=0.2
)

#scaling down the pixel values of each image by a factor of 1/255 to normalizes the image data between 0 and 1 for test data.
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

#generating batches of training data
train_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(120, 120),#resizing image to 120*120
    color_mode='rgb', #converting image to RGB format
    class_mode='binary',#only 2 classes, crack and noncrack
    batch_size=16, #number of image to include in each batch
    shuffle=True, #suffling data before each epoch
    seed=42,
    subset='training'
)

#generating batches of validation data
'''val_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath', #column containing filepath of images
    y_col='Label', #column containing label of images
    target_size=(120, 120),
    color_mode='rgb',
    class_mode='binary',
    batch_size=16,
    shuffle=True,
    seed=42,
    subset='validation'
)'''

#generating batches of test data
test_data = train_gen.flow_from_dataframe(
    test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(120, 120),
    color_mode='rgb',
    class_mode='binary',
    batch_size=16,
    shuffle=False,
    seed=42
)

#defining the architecture of the Convolution neural network (CNN) model
inputs = tf.keras.Input(shape=(120, 120, 3)) #input layer with 3-channel images
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs) #applying convulation layer of 16 filter, each of size 3*3 pixel and ReLU activation layer to input
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x) #reducing the spatial size of the output
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x) #reducing the spatial dimension of the output to single value
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x) #dense layer with a single output and sigmoid activation function

#building the model by using the architecture of the CNN 
model = tf.keras.Model(inputs=inputs, outputs=outputs)


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
) #configuring learning process of the model by giving optimizer, loss and metrices.

print(model.summary()) 

history = model.fit(
    train_data,
    #validation_data=val_data,
    epochs=12) #training model on training data


#defining the function to evaluate the model on testing data
def evaluate_model(model, test_data):
    
    results = model.evaluate(test_data, verbose=0)
    loss = results[0]
    acc = results[1]
    
    print("    Test Loss: {:.5f}".format(loss))
    print("Test Accuracy: {:.2f}%".format(acc * 100))
    
evaluate_model(model, test_data) #evaluating the testing data

# Get the predicted probabilities for the test data
y_pred = model.predict(test_data).ravel()

# Compute the false positive rate, true positive rate, and threshold
fpr, tpr, thresholds = roc_curve(test_data.classes, y_pred)

# Compute the area under the ROC curve
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.plot(fpr, tpr, lw=1, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random guess')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# Define the image size
IMG_SIZE = 120

# Define the labels
labels = {0: 'nocrack', 1: 'crack'}

# Define the path to the folder containing images to predict
folder_path = 'C:/Users/14098/OneDrive - Lamar University/Desktop/Image/extrapredict'

# Get a list of all the image file names in the folder
image_files = os.listdir(folder_path)

# Loop over all the images and make predictions
for image_file in image_files:
    # Load the image
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)

    # Preprocess the image
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    # Make the prediction
    pred = model.predict(image)
    pred_class = int(np.round(pred[0][0]))

    # Get the label of the predicted class
    label = labels[pred_class]

    # Display the image and prediction
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title('Prediction: {}'.format(label))
    plt.show()
