import numpy as np
import tensorflow as tf
import timeit
import dataloader as dl
import models
import lrsched

EPOCH_COUNT = 90
BATCH_SIZE = 125
VAL_BATCH_SIZE = 125
TRAIN_LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
IMG_WIDTH = 256#224
IMG_HEIGHT = 256#224
IMG_IN_WIDTH = 227
IMG_IN_HEIGHT = 227
IMG_CHANNELS = 3

timeStart = timeit.default_timer()

np.random.seed(0)

dataDir = '../data'
trainDataDir = f'{dataDir}/ILSVRC2012_img_train'
valDataDir = f'{dataDir}/ILSVRC2012_img_val'
mappingFile = f'{dataDir}/ILSVRC2012_mapping.txt'
valLabels = f'{dataDir}/ILSVRC2012_validation_ground_truth.txt'
logFile = f'{dataDir}/log.csv'

with open(mappingFile) as f:
	mapping = f.readlines()

classNames = [line.split(' ')[1][:-1] for line in mapping]

trainDs = tf.keras.utils.image_dataset_from_directory(trainDataDir, class_names = classNames, seed = 0, image_size = (IMG_HEIGHT, IMG_WIDTH), batch_size = BATCH_SIZE)

trainLoader = dl.DataLoader(trainDs, IMG_WIDTH, IMG_HEIGHT, IMG_IN_WIDTH, IMG_IN_HEIGHT, IMG_CHANNELS)
trainGen = trainLoader.generator(augment = True)

with open(valLabels) as f:
	labels = f.readlines()

labels = [int(label[:-1]) - 1 for label in labels]

valDs = tf.keras.utils.image_dataset_from_directory(valDataDir, labels = labels, seed = 0, image_size = (IMG_HEIGHT, IMG_WIDTH), batch_size = VAL_BATCH_SIZE)

valLoader = dl.DataLoader(valDs, IMG_WIDTH, IMG_HEIGHT, IMG_IN_WIDTH, IMG_IN_HEIGHT, IMG_CHANNELS)
valGen = valLoader.generator(augment = False)

model = models.AlexNet(IMG_WIDTH, inShape = (IMG_IN_WIDTH, IMG_IN_HEIGHT, IMG_CHANNELS), batchSize = BATCH_SIZE)

model.build(input_shape = (None, IMG_HEIGHT, IMG_HEIGHT, IMG_CHANNELS))

model.compile(loss = 'categorical_crossentropy', optimizer = tf.optimizers.SGD(learning_rate = TRAIN_LR, momentum = MOMENTUM, weight_decay = WEIGHT_DECAY), metrics = ['accuracy'])

model.summary()

reduceLR = lrsched.ReduceLROnPlateau(factor = 0.1)

csvLogger = tf.keras.callbacks.CSVLogger(logFile, append = True, separator = ';')

model.fit(epochs = EPOCH_COUNT, x = trainGen, validation_data = valGen, validation_freq = 1, callbacks = [reduceLR, csvLogger])

timeStop = timeit.default_timer()

timeElapsed = timeStop - timeStart

print(f'Time elapsed: {timeElapsed}')
