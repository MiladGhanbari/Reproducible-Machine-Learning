from __future__ import print_function

import os.path

import densenet_C
import math
import numpy as np
import sklearn.metrics as metrics

from keras.datasets import cifar10
from keras.utils import np_utils, multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, CSVLogger
from keras import backend as K

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

batch_size = 64
nb_classes = 10
nb_epoch = 100

img_rows, img_cols = 32, 32
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = -1
dropout_rate = 0.0 # 0.0 for data augmentation

model = densenet_C.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block, bottleneck=False, reduction=0.0,
                          growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None)
#model = multi_gpu_model(model, gpus=2)
print("Model created")

model.summary()

#optimizer = Adam(lr=1e-1) # Using Adam instead of SGD to speed up training
optimizer = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("Finished compiling")
print("Building model...")

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
testX = testX.astype('float32')

trainX = densenet_C.preprocess_input(trainX)
testX = densenet_C.preprocess_input(testX)

Y_train = np_utils.to_categorical(trainY, nb_classes)
Y_test = np_utils.to_categorical(testY, nb_classes)

generator = ImageDataGenerator(width_shift_range=5./32,
                               height_shift_range=5./32,
                               horizontal_flip=True)
'''
generator = ImageDataGenerator(width_shift_range=0,
                               height_shift_range=0,
                               horizontal_flip=False)
'''
generator.fit(trainX, seed=0)

# Load model
#weights_file="weights/DenseNet-13-08-CIFAR10.h5"
#if os.path.exists(weights_file):
#   #model.load_weights(weights_file, by_name=True)
#    print("Model loaded.")

out_dir_weight = "weights/approach_C.h5"
out_dir_logs = "logs/approach_C.log"

def step_decay(epoch):
    
    init_LR = 0.1
    decay = 0
    #lr = init_LR * math.pow(0.5    
    if epoch >= 100 * 0.75:
        decay = 2 
    elif epoch >= 100 * 0.5:
        decay = 1

    
    return init_LR * math.pow(0.1, decay)

lrate = LearningRateScheduler(step_decay)



#lr_reducer      = ReduceLROnPlateau(monitor='val_acc', factor=0.75,
                                    #cooldown=0, patience=5, min_lr=1e-5)
model_checkpoint= ModelCheckpoint(out_dir_weight, monitor="val_acc", save_best_only=True,
                                  save_weights_only=True, verbose=1)

csv_logger = CSVLogger(out_dir_logs, append=True)

callbacks=[lrate, model_checkpoint, csv_logger]

model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size),
                    steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
                    callbacks=callbacks,
                    validation_data=(testX, Y_test),
                    validation_steps=testX.shape[0] // batch_size, verbose=1)

yPreds = model.predict(testX)
yPred = np.argmax(yPreds, axis=1)
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)

