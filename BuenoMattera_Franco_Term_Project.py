from __future__ import print_function
from numpy.random import seed
from datetime import timedelta, datetime
import multiprocessing
from multiprocessing import process
import time
import re
import sys
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, CountVectorizer,IDF
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml import Pipeline
from datetime import time
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.image import _ImageSchema, ImageSchema
from pyspark.ml.linalg import SparseVector, DenseVector
from pyspark.ml.linalg import Vectors, VectorUDT
import os
import matplotlib.pyplot as plt
from matplotlib import image
from numpy import rot90
from PIL import Image
from pyspark.sql.functions import when
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from pyspark.sql.functions import udf
import pandas as pd
from pyspark.sql.types import FloatType 
from pyspark.ml.feature import UnivariateFeatureSelector
from pyspark.ml import Pipeline
from datetime import time
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.classification import MultilayerPerceptronClassifier, RandomForestClassifier, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from numpy import source
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras import Input, Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.metrics import CategoricalAccuracy, BinaryAccuracy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()

########################################################################################## 1. Image Preprocessing ################################################################################################################


# Load File:

train_label = spark.read.csv('archive/train_labels.csv', header=True)

# Filtering Dataset to obtain relevant species:
filtered_train = train_label.filter((train_label.label == 'Lithraea_caustica') | (train_label.label == 'Peumus_boldus') | (train_label.label == 'Ulmus_americana')).sort(train_label.label.asc()) # ordered alphabetically by label
# Source: https://www.datasciencemadesimple.com/subset-or-filter-data-with-multiple-conditions-in-pyspark/

# Creating List of images: The commented lines below only run once
list_of_train_images = [i[0] for i in filtered_train.collect()]
# copy images to new folder
#for i in list_of_train_images:
#    print(i)
#    os.system("cp archive/train_images/{} archive/filtered_train_images".format(i)) 
    #Source: https://stackoverflow.com/questions/4256107/running-bash-commands-in-python



def rename_pics(list_of_images = list_of_train_images, source ='archive/filtered_train_images/', prefix = 'train_0000000', extension = '.jpg'):

    """This function renames the images so the are in order.
    It should only run once and after is applied, the order of the labels should match the files """
    
    for i, j in enumerate(list_of_images):
        os.rename(os.path.join(source,j),os.path.join(source, prefix+str(i)+extension))
    #Source: https://stackoverflow.com/questions/63745780/system-cannot-find-the-file-specified-when-renaming-image-files

#rename_pics() # no w all files are  a;readyordered with the label

list_of_filtered_images = sorted(os.listdir('archive/filtered_train_images/'))[1:] # run once!


#list_of_train_images = os.listdir('archive/train_images/')

def rotating_images(list_of_images = list_of_filtered_images, zeroes_prefix = '0000000'):
    
    """This Function loops trough all images and rotate them 3 times in order to give 
    The algorithm more data to train on"""
    #Loop trough all files
    
    #opening all images one by one:

    im_count = len(list_of_images)
    
    for image in list_of_images:

        im_count += 3
        tree_image = plt.imread('archive/filtered_train_images/{}'.format(image))  # this will transform the image into an array 
        rot90_label = zeroes_prefix + str(im_count-3)
        tree_image_rot90 = plt.imsave('archive/filtered_train_images/train_{}.jpg'.format(rot90_label), rot90(tree_image))
        rot_180_label = zeroes_prefix + str(im_count-2)
        tree_image_rot180 = plt.imsave('archive/filtered_train_images/train_{}.jpg'.format(rot_180_label), rot90(tree_image, k = 2))
        rot_270_label = zeroes_prefix + str(im_count-1)
        tree_image_rot270 = plt.imsave('archive/filtered_train_images/train_{}.jpg'.format(rot_270_label), rot90(tree_image, k = 3))
        print("Loaded: train_{}.jpg, train_{}.jpg, train_{}.jpg images".format(rot90_label, rot_180_label, rot_270_label))
    print('Load Complete')

#Source: https://www.pythonpool.com/python-loop-through-files-in-directory/
#https://www.moonbooks.org/Articles/How-to-rotate-and-plot-a-RGB-image-with-scipy-and-matplotlib-in-python/

#rotating_images() # all files already loaded. This should also run once


#add rest of labels to dataset: concatenate itself 3 times
filtered_train1 = filtered_train.union(filtered_train)
filtered_train2 = filtered_train1.union(filtered_train)
filtered_train_extended= filtered_train2.union(filtered_train)

#Source: https://sparkbyexamples.com/pyspark/pyspark-read-csv-file-into-dataframe/

list_of_filtered_images_extended = sorted(os.listdir('archive/filtered_train_images/'))[1:] # run once per update!


def load_img_to_rdd2(list_of_images = list_of_filtered_images_extended, img_folder = 'filtered_train_images', dataset_rdd = filtered_train_extended.rdd.collect()):

    """This function will create a dataset with roid, features, filename and original label"""
    
    dataset = []
    rowid = 0
    for image, info in zip(list_of_images, dataset_rdd):
        rowid +=1 
        train_imgage = Image.open('archive/{}/{}'.format(img_folder,image)).resize((64,64))
        imgArray = (img_to_array(train_imgage).ravel())
        imgArray = Vectors.dense(imgArray)
        filename = info[0] # original filename
        label = info[1]
        dataset.append((rowid, imgArray, filename,label))

    return(dataset)

############################### Train Set ##################################:
train_df = load_img_to_rdd2()
df = spark.createDataFrame(train_df, ["rowid", "features", 'filename', 'original_label'])

train_df = [] # to release memory 
# Adding numeric label:
df = df.withColumn('label', when(df.original_label == 'Lithraea_caustica', float(1)).when(df.original_label == 'Peumus_boldus', float(2)).when(df.original_label == 'Ulmus_americana', float(3)))
# Source: https://sparkbyexamples.com/pyspark/pyspark-when-otherwise/

# Normalizing features:
rdd_train = df.rdd.map(lambda x: (x[0], x[1]/255, x[2], x[3], x[4])) 
df = rdd_train.toDF(["rowid", "features", 'filename', 'original_label', 'label'])


############################### Test Set ##################################:
test_label = spark.read.csv('archive/test_labels.csv', header=True)
filtered_test = test_label.filter((test_label.label == 'Lithraea_caustica') | (test_label.label == 'Peumus_boldus') | (test_label.label == 'Ulmus_americana')).sort(test_label.label.asc()) # ordered alphabetically by label

# Creating List of images
list_of_test_images = [i[0] for i in filtered_test.collect()]
# copy images to new folder
#for i in list_of_test_images:
#    print(i)
#    os.system("cp archive/test_images/{} archive/filtered_test_images".format(i)) 
    #Source: https://stackoverflow.com/questions/4256107/running-bash-commands-in-python

#rename_pics(list_of_images = list_of_test_images, source ='archive/filtered_test_images/', prefix = 'test_0000000', extension = '.jpg')
new_list_test_img = sorted(os.listdir('archive/filtered_test_images/'))[1:] # filter hidden files

test_df = load_img_to_rdd2(list_of_images = new_list_test_img,img_folder='filtered_test_images', dataset_rdd = filtered_test.rdd.collect())
df_test = spark.createDataFrame(test_df, ["rowid", "features", "filename", 'original_label'])
df_test = df_test.withColumn('label', when(df_test.original_label == 'Lithraea_caustica', float(1)).when(df_test.original_label == 'Peumus_boldus', float(2)).when(df_test.original_label == 'Ulmus_americana', float(3)))

# Normalizing features:
rdd_test = df_test.rdd.map(lambda x: (x[0], x[1]/255, x[2], x[3], x[4])) 
df_test = rdd_test.toDF(["rowid", "features", "filename", 'original_label', 'label'])

number_of_features = len(df.rdd.take(1)[0][1]) # 12288


############################################ First Approach, normal Model tunning ################################################################

####  1.  Training the model using native ml spark:

mlp = MultilayerPerceptronClassifier(layers=[12288, 10, 4], seed=90, blockSize=32, stepSize=0.03).setMaxIter(80) 
mlp.setFeaturesCol("features") 
model = mlp.fit(df)
predictions = model.transform(df_test)

################################# Predictions #################################:
# 
true_vs_predictions = predictions.select("label","prediction").rdd.map(lambda x: (float(x[0]), float(x[1])))
metrics = MulticlassMetrics(true_vs_predictions)
confusion_matrix = metrics.confusionMatrix().toArray()
Precision = metrics.weightedPrecision
Recall = metrics.weightedRecall
F1 = metrics.weightedFMeasure()
# Source: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.MultilayerPerceptronClassifier.html

print("\nPerformance Metrics: Multilayered Perceptron Classifier - Mmanual Tunning")
print("Precision:", Precision)
print("Recall:", Recall)
print("F1:", F1)
print("Confusion Matrix:", confusion_matrix)

 #Sources: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.clustering.LDA.html?highlight=lda#pyspark.ml.clustering.LDA
    # https://github.com/trajanov/BigDataAnalytics/blob/master/Notebooks/Spark-Example-22-LDA.ipynb



####  2.  Training the model using native Gradient Boosting Classifier:

from pyspark.ml.classification import GBTClassifier

nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
nb_model = nb.fit(df)
predictions_nb = nb_model.transform(df_test)
predictions_nb.count()

# source: https://spark.apache.org/docs/latest/ml-classification-regression.html#naive-bayes

################################# Predictions #################################:
# 

true_vs_predictions_nb = predictions_nb.select("label","prediction").rdd.map(lambda x: (float(x[0]), float(x[1])))
metrics_nb = MulticlassMetrics(true_vs_predictions_nb)
confusion_matrix_nb = metrics_nb.confusionMatrix().toArray()
Precision_nb = metrics_nb.weightedPrecision
Recall_nb = metrics_nb.weightedRecall
F1_nb = metrics_nb.weightedFMeasure()

print("\nPerformance Metrics: Naive Bayes Classifier - Mmanual Tunning")
print("Precision:", Precision_nb)
print("Recall:", Recall_nb)
print("F1:", F1_nb)
print("Confusion Matrix:", confusion_matrix_nb)

# 3. Approximation to big pre- trained networks :
# Approach was unnsuccessful: Computation requirements are greater

#mlp2 = MultilayerPerceptronClassifier(layers=[12288,3072,4], seed=2, blockSize=128, stepSize=0.02).setMaxIter(80) 
#mlp2.setFeaturesCol("features") 
#model2 = mlp2.fit(df)
#predictions2 = model2.transform(df_test)
#predictions2.count()
################################# Predictions #################################:
# 
#true_vs_predictions2 = predictions2.select("label","prediction").rdd.map(lambda x: (float(x[0]), float(x[1])))
#metrics2 = MulticlassMetrics(true_vs_predictions2)
#confusion_matrix2 = metrics2.confusionMatrix().toArray()
#Precision2 = metrics2.weightedPrecision
#Recall2 = metrics2.weightedRecall
#F1_2 = metrics2.weightedFMeasure()

#print("\nPerformance Metrics: Multilayered Perceptron Classifier - Mmanual Tunning")
#print("Precision:", Precision2)
#print("Recall:", Recall2)
#print("F1:", F1_2)
#print("Confusion Matrix:", confusion_matrix2)

############################# Comparison With Regular Transfer Learning ################################


# Creating List of images
list_of_train_images = [i[0] for i in filtered_train.collect()]
list_of_test_images = [i[0] for i in filtered_test.collect()]
# copy images to new folder

def copy_img(option = 'Train'):

    '''This function copy files to a second folder'''

    if option == 'Train':

        for i in list_of_train_images:
            print(i)
            os.system("cp archive/train_images/{} archive/filtered_train_images2".format(i)) 
    elif option == 'Test':
        for i in list_of_test_images:
            print(i)
            os.system("cp archive/test_images/{} archive/filtered_test_images2".format(i)) 
            #https://stackoverflow.com/questions/4256107/running-bash-commands-in-python

#copy_img(option='Test')
#copy_img()

from numpy import source


def rename_pics(list_of_images = list_of_train_images, source ='archive/filtered_train_images2/', filtered_data = filtered_train.rdd.collect(), extension = '.jpg'):
    # renaming existing images: # it only runs once and now the order of pictures its from o to max matchin the order of the labels
    """This Function labels images in compliance with library requirements """
    for i, j, k in zip(range(0, len(list_of_images)), list_of_images, filtered_data):
        if k[1] == 'Lithraea_caustica':
            os.rename(os.path.join(source,j),os.path.join(source, 'Lithraea_caustica '+'000'+str(i)+extension))
        elif k[1] == 'Peumus_boldus':
            os.rename(os.path.join(source,j),os.path.join(source, 'Peumus_boldus '+'000'+str(i)+extension))
        else:
            os.rename(os.path.join(source,j),os.path.join(source, 'Ulmus_americana '+'000'+str(i)+extension))
    #https://stackoverflow.com/questions/63745780/system-cannot-find-the-file-specified-when-renaming-image-files

#rename_pics() # no w all files are  aready ordered with the label
#rename_pics(list_of_images = list_of_test_images, source ='archive/filtered_test_images2/', filtered_data = filtered_test.rdd.collect(), extension = '.jpg')

list_of_filtered_images = sorted(os.listdir('archive/filtered_train_images2/'))[1:] # run once!

def image_process(path = 'archive/filtered_train_images2/'):

    """This function uses keras/tensofrlow API to grab and preprocess images and return a generator"""
    
    datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, shear_range=0.25, zoom_range=0.25)
    tset = datagen.flow_from_directory(path, target_size=(128, 128), batch_size=32, class_mode='categorical')
    return(tset)

# Source: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator, Udemy: Deep-Learning A-z

train_images_array = image_process()

test_images_array = image_process(path = 'archive/filtered_test_images2/')





##### Using Blueprint API from Keras/tf official Documentation: https://keras.io/guides/transfer_learning/
# Works for any pretrained avaiable on library

##udf ----> more research is needed
def MLP(): # no seed available

    vg_model = VGG16(weights="imagenet", input_shape=(300, 300, 3),include_top=False)
    vg_model.trainable = False
    inputs = Input(shape=(300, 300, 3))
    x = vg_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(3)(x)
    model = Model(inputs, outputs)

    model.compile(optimizer=Adam(),loss=CategoricalCrossentropy(from_logits=True),metrics=[CategoricalAccuracy()])
    model.fit(train_images_array, epochs=35)
    return model
    
#Additional ref: https://towardsdatascience.com/transfer-learning-with-vgg16-and-keras-50ea161580b4

trained_model = MLP()

def vg_16_predictions():

    preds = trained_model.predict(test_images_array)

    for i in range(0, len(preds)):
        if preds[i][0] >= 0.5:
            preds[i][0] = 1.0
        else:
            preds[i][0] = 0.0
        if preds[i][1] >= 0.5:
            preds[i][1] = 1.0
        else:
            preds[i][1] = 0.0
        if preds[i][2] >= 0.5:
            preds[i][2] = 1.0
        else:
            preds[i][2] = 1.0
    final_pred = []
    for i in range(0, len(preds)):
        if preds[i][0] == 1.0:
            final_pred.append(1.0)
        elif preds[i][1] == 1.0:
            final_pred.append(2.0)
        elif preds[i][2] == 1.0:
            final_pred.append(3.0)
    
    test_labels_array = test_images_array[0][1]

    final_test_labels = []
    for i in range(0, len(test_labels_array)):
        if test_labels_array[i][0] == 1.0:
            final_test_labels.append(1)
        elif test_labels_array[i][1] == 1.0:
            final_test_labels.append(2)
        elif test_labels_array[i][2] == 1.0:
            final_test_labels.append(3)


    #confusion_matrix_vg16 = confusion_matrix(final_test_labels, final_pred)
    Precision_vg16 = precision_score(final_test_labels, final_pred, average = 'weighted')
    Recall_vg16 = recall_score(final_test_labels, final_pred,average = 'weighted' )
    F1_vg16 = f1_score(final_test_labels, final_pred, average = 'weighted')

    print("\nPerformance Metrics: Vg-16: ")
    print("Precision:", Precision_vg16)
    print("Recall:", Recall_vg16)
    print("F1:", F1_vg16)
    #print("Confusion Matrix:", confusion_matrix_vg16)
#Reference: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

vg_16_predictions()































































