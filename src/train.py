from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import multi_gpu_model
from keras import optimizers as optims
from keras import regularizers as regs
from keras.models import Model
from sir_utils import gen_from_directory_labelled
import tensorflow as tf
import keras.backend as K
import os

#config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4, \
#                                allow_soft_placement=True, device_count = {'CPU': 1})
#session = tf.Session(config=config)
#K.set_session(session)

USE_GPU = True
NUM_GPUS = 4
TRAINABLE_BASE = True
#opt = optims.SGD(lr=0.00001, decay=1e-7, momentum=0.9)
opt = optims.Adam(lr=0.000001, decay=1e-7)
NUM_CLASSES = 23
STEPS = 2048
EPOCHS = 32
EARLY_STOPPING = EarlyStopping(monitor='val_loss', patience=3)
CHECKPOINT = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=16)
MULTILABEL = False
MODEL_PATH = '/mnt/anindya/models/'
LOAD_WEIGHTS = True
INITIAL_EPOCH = 0

if MULTILABEL:
    METRICS=['categorical_accuracy', 'accuracy']
    LOSS='binary_crossentropy'
    ACTIVATION = 'sigmoid'
    MODEL_SUFFIX = '_multilabel.h5'
else:
    METRICS=['categorical_accuracy']
    LOSS='categorical_crossentropy'
    ACTIVATION = 'softmax'
    MODEL_SUFFIX = '_unilabel_all.h5'

train_gen = gen_from_directory_labelled('/mnt/anindya/out/', '/mnt/anindya/items_interests.txt', NUM_GPUS*32, MULTILABEL)
test_gen = gen_from_directory_labelled('/mnt/anindya/out_test/', '/mnt/anindya/items_interests.txt', NUM_GPUS*32, MULTILABEL)

base_models = {
        #'inceptionv3': InceptionV3(weights='imagenet', pooling='avg', include_top=False),
        'inceptionresnetv2': InceptionResNetV2(weights='imagenet', pooling='avg', include_top=False),
        'resnet50': ResNet50(weights='imagenet', pooling='avg', include_top=False),
        #'mobilenet': MobileNet(weights='imagenet', pooling='avg', include_top=False),
        #'xception': Xception(weights='imagenet', pooling='avg', include_top=False),
        #'vgg19': VGG19(weights='imagenet', pooling='avg', include_top=False),
        #'vgg16': VGG16(weights='imagenet', pooling='avg', include_top=False)
        }

def run_model(model_name, base):
    model_file = MODEL_PATH + model_name + MODEL_SUFFIX
    x = base.output
    x = BatchNormalization(axis=1)(x)
    predictions = Dense(NUM_CLASSES, kernel_initializer='he_normal', kernel_regularizer=regs.l2(0.01), activation=ACTIVATION)(x)

    model = Model(inputs=base.input, outputs=predictions)
    for layer in base.layers:
        layer.trainable = TRAINABLE_BASE

    if USE_GPU:
        final = multi_gpu_model(model, gpus=NUM_GPUS)
    else:
        final = model

    final.compile(optimizer=opt, loss=LOSS, metrics=METRICS)
    print 'model compiled'
    if LOAD_WEIGHTS and os.path.isfile(model_file):
        model.load_weights(model_file)
    print 'weights loaded'
    final.fit_generator(
            train_gen,
            steps_per_epoch=STEPS,
            epochs=EPOCHS,
            callbacks=[EARLY_STOPPING],
            validation_data=test_gen,
            validation_steps=4,
            max_queue_size=20,
            initial_epoch=INITIAL_EPOCH)
    model.save_weights(model_file)
    print 'FINISHED=====================' + model_name + '================='

for n, b in base_models.items():
    print n
    print '======================'
    run_model(n, b)
