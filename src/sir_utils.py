from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, BatchNormalization
from keras.utils import multi_gpu_model
from keras import optimizers as optims
from keras import regularizers as regs
from keras.models import Model
from glob import glob
import os
import shutil
import pandas as pd
import numpy as np
import cv2

KERAS_BASES = {
    'inceptionv3': InceptionV3(weights='imagenet', pooling='avg', include_top=False),
    'inceptionresnetv2': InceptionResNetV2(weights='imagenet', pooling='avg', include_top=False),
    'resnet50': ResNet50(weights='imagenet', pooling='avg', include_top=False),
    'xception': Xception(weights='imagenet', pooling='avg', include_top=False),
    'vgg19': VGG19(weights='imagenet', pooling='avg', include_top=False),
    'vgg16': VGG16(weights='imagenet', pooling='avg', include_top=False)
    }

def num_batches_in_dir(path, batch_sz=128):
    return np.floor(len(glob(path+'/*'))/batch_sz).astype('int') + 1

def get_image_data(path):
    img = cv2.imread(path)[np.newaxis, :].astype('float32')
    img /= 255.
    img -= 0.5
    img *= 2.
    return img

get_batch_data = lambda path, files: np.vstack([get_image_data(path+'/'+str(_)+'.jpg') for _ in files.astype('int')]).astype('float32')

def gen_from_directory_labelled(path, key, batch_size=128, multilabel=False):
    x = pd.read_csv(key)
    items = pd.DataFrame([int(_.split('/')[-1].split('.')[0]) for _ in glob(path+'/*')], columns=['item_id']).astype('int')
    x = x.merge(items)
    if multilabel:
        x = x.pivot(index='item_id', columns='interest_id', values='values').fillna(0).reset_index().values.astype('float32')
        y = x.values[:, 1:]
    else:
        over = set(x.item_id.value_counts().keys()[x.item_id.value_counts().values > 1])
        x = x[~x.item_id.isin(over)]
        y = pd.get_dummies(x['interest_id'].values).values.astype('float32')
    x = x['item_id'].values
    start = 0
    while True:
        if start+batch_size < x.shape[0]:
            end = start + batch_size
            yield (get_batch_data(path, x[start:end]), y[start:end, :])
            start = end
        else:
            end = x.shape[0]
            yield (get_batch_data(path, x[start:end]), y[start:end, :])
            start = 0

def gen_from_directory_unlabelled(path, batch_size=128):
    items = np.array([int(_.split('/')[-1].split('.')[0]) for _ in glob(path+'/*')])
    start = 0
    while start < len(items):
        end = min(start+batch_size, len(items))
        yield items[start:end].reshape(-1, 1), get_batch_data(path, items[start:end])
        start = end

def get_snupps_model(base, num_classes=23, activation='softmax'):
    x = base.output
    x = BatchNormalization(axis=1)(x)
    x = Dense(num_classes, kernel_initializer='he_normal', kernel_regularizer=regs.l2(0.01), activation=activation)(x)
    model = Model(inputs=base.input, outputs=x)
    return model

def configure_model(model, weight_path='', optimizer=optims.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['categorical_accuracy'], num_gpus=4):
    if num_gpus > 0:
        final = multi_gpu_model(model, gpus=num_gpus)
    else:
        final = model

    final.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    if len(weight_path) > 0:
        model.load_weights(weight_path)

    return final

def detach_model(m):
    for l in m.layers:
        if l.name == 'model_1':
            return l
    return m

def save_model_weights(model, path):
    m = detach_model(model)
    m.save_weights(path)

def get_keras_base(name):
    return KERAS_BASES[name]

def train_update_model(name, data, key, steps_per_epoch=512, epochs=32, weight_path=''):
    m = get_snupps_model(get_keras_base(name))
    m = configure_model(model=m, weight_path=weight_path)
    gen = gen_from_directory_labelled(data, key)
    m.fit_generator(
            gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs)
    return m

def evaluate_model(model, data, key, num_batches=16):
    gen = gen_from_directory_labelled(data, key)
    return model.evaluate_generator(gen, steps=16)

def predict_model(model, data_path):
    gen = gen_from_directory_unlabelled(data_path)
    return model.predict_generator(gen, steps=num_batches_in_dir(data_path))

def get_encodings(data_path, model, layer_idx=-2, batch_size=128):
    gen = gen_from_directory_unlabelled(data_path, batch_size)
    dm = detach_model(model)
    dm = Model(inputs=dm.layers[0].input, outputs=dm.layers[layer_idx].output)
    preds = [np.hstack([names, dm.predict_on_batch(batch)]) for names, batch in gen]
    return np.vstack(preds)

def process_images(inpath, outpath, shape=(299, 299)):
    def store_resized(pic):
        img = cv2.imread(inpath+pic)
        img = cv2.resize(img, shape)
        cv2.imwrite(outpath+pic, img)
        
    num_cpus = cpu_count()
    infiles = [_.split('/')[-1] for _ in glob(inpath+'*')]
    donefiles = [_.split('/')[-1] for _ in glob(outpath+'*')]
    freshfiles = set(infiles) - set(donefiles)
    p = Pool(num_cpus)
    p.map(store_resized, freshfiles, num_cpus)
    p.map(os.remove, [inpath+_ for _ in fresh], num_cpus)

def copy_file(src_dest_tuple):
    shutil.copy(src_dest_tuple[0], src_dest_tuple[1])

def segregate_files_by_category(inpath, outpath, key_file):
    key = pd.read_csv(key_file)
    f = [int(_.split('/')[-1].split('.')[0]) for _ in glob(inpath+'*')]
    key = key[key.item_id.isin(set(f))].sort_values('interest_id')
    labels = [str(_) for _ in key.interest_id.unique()]
    for l in labels:
        if not os.path.exists(outpath+l):
            os.makedirs(outpath+l)
    moves = [(inpath+str(_[0])+'.jpg', outpath+str(_[1])+'/') for _ in key.to_records(index=False)]
    p = Pool(cpu_count())
    p.map(copy_file, moves, cpu_count())

