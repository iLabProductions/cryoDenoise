
import numpy as np

import tensorflow as tf

from settings import (TF_RECORDS, BATCH_SIZE, BUFFER_SIZE)


def create_list_dataset(data_dir):
    return tf.data.Dataset.list_files(data_dir, shuffle=False)

def parse_func(record):
    ''' NRH: CUSTOM PARSE FUNCTION
    '''

    features = { 'input' : tf.io.FixedLenFeature([], dtype=tf.string),
                 'target': tf.io.FixedLenFeature([], dtype=tf.string)}

    # read in and decode
    record = tf.io.parse_single_example(record, features=features)

    input    = tf.io.decode_raw(record['input'], tf.float32)
    target   = tf.io.decode_raw(record['target'], tf.float32)


    input=tf.cast(input,tf.float32)
    target=tf.cast(target,tf.float32)

    input  = tf.reshape(input, [96, 96, 96, -1])
    target = tf.reshape(target, [96, 96, 96, -1])


    #Rotation augment


    #input  = tf.image.per_image_standardization(input)
    #target = tf.image.per_image_standardization(target)

    # input = _standardization(input)
    # target = _standardization(target)

    # transfer 1 channel -> 3 identical channels
    input  = tf.transpose(input, [3, 0, 1,2])
    target = tf.transpose(target, [3, 0, 1, 2])


    # #Rotation Augment
    a,b=tf.split(tf.random.uniform([2],maxval=4,dtype=tf.int32),num_or_size_splits=2)
    a=tf.squeeze(a)
    b=tf.squeeze(b)

    input=tf.image.rot90(input,a)
    input=tf.transpose(tf.image.rot90(tf.transpose(input,[0,1,3,2]),b),[0,1,3,2])

    target=tf.image.rot90(target,a)
    target=tf.transpose(tf.image.rot90(tf.transpose(target,[0,1,3,2]),b),[0,1,3,2])

    input  = tf.transpose(input, [1, 2, 3, 0])
    target = tf.transpose(target, [1, 2, 3, 0])

    original=target

    return (original, input, target)


def parse_func_new(record):
    ''' NRH: CUSTOM PARSE FUNCTION
    '''

    features = { 'input' : tf.io.FixedLenFeature([], dtype=tf.string),
                 'target': tf.io.FixedLenFeature([], dtype=tf.string),
                 'label' : tf.io.FixedLenFeature([], dtype=tf.string)}

    # read in and decode
    record = tf.io.parse_single_example(record, features=features)

    input    = tf.io.decode_raw(record['input'], tf.float32)
    target   = tf.io.decode_raw(record['target'], tf.float32)
    label    = tf.io.decode_raw(record['label'], tf.float32)


    input=tf.cast(input,tf.float32)
    target=tf.cast(target,tf.float32)

    input  = tf.reshape(input, [96, 96, 96, -1])
    target = tf.reshape(target, [96, 96, 96, -1])


    #Rotation augment


    #input  = tf.image.per_image_standardization(input)
    #target = tf.image.per_image_standardization(target)

    # input = _standardization(input)
    # target = _standardization(target)

    # transfer 1 channel -> 3 identical channels
    input  = tf.transpose(input, [3, 0, 1,2])
    target = tf.transpose(target, [3, 0, 1, 2])


    # #Rotation Augment
    a,b=tf.split(tf.random.uniform([2],maxval=4,dtype=tf.int32),num_or_size_splits=2)
    a=tf.squeeze(a)
    b=tf.squeeze(b)

    input=tf.image.rot90(input,a)
    input=tf.transpose(tf.image.rot90(tf.transpose(input,[0,1,3,2]),b),[0,1,3,2])

    target=tf.image.rot90(target,a)
    target=tf.transpose(tf.image.rot90(tf.transpose(target,[0,1,3,2]),b),[0,1,3,2])

    input  = tf.transpose(input, [1, 2, 3, 0])
    target = tf.transpose(target, [1, 2, 3, 0])

    original=target

    return (original, input, target)


def parse_func_label(record):
    ''' NRH: CUSTOM PARSE FUNCTION
    '''

    features = { 'input' : tf.io.FixedLenFeature([], dtype=tf.string),
                 'target': tf.io.FixedLenFeature([], dtype=tf.string),
                 'label' : tf.io.FixedLenFeature([], dtype=tf.string)}

    # read in and decode
    record = tf.io.parse_single_example(record, features=features)

    input    = tf.io.decode_raw(record['input'], tf.float32)
    target   = tf.io.decode_raw(record['target'], tf.float32)
    label    = record['label']

    return label

def split_dataset(dataset: tf.data.Dataset, validation_data_fraction: float):
    """
    Splits a dataset of type tf.data.Dataset into a training and validation dataset using given ratio. Fractions are
    rounded up to two decimal places.
    @param dataset: the input dataset to split.
    @param validation_data_fraction: the fraction of the validation data as a float between 0 and 1.
    @return: a tuple of two tf.data.Datasets as (training, validation)
    """

    validation_data_percent = round(validation_data_fraction * 100)
    if not (0 <= validation_data_percent <= 100):
        raise ValueError("validation data fraction must be ∈ [0,1]")

    dataset = dataset.enumerate()
    train_dataset = dataset.filter(lambda f, data: f % 100 > validation_data_percent)
    validation_dataset = dataset.filter(lambda f, data: f % 100 <= validation_data_percent)

    # remove enumeration
    train_dataset = train_dataset.map(lambda f, data: data)
    validation_dataset = validation_dataset.map(lambda f, data: data)

    return train_dataset, validation_dataset

def split_dataset_new(dataset: tf.data.Dataset, validation_data_fraction: float):
    """
    Splits a dataset of type tf.data.Dataset into a training and validation dataset using given ratio. Fractions are
    rounded up to two decimal places.
    @param dataset: the input dataset to split.
    @param validation_data_fraction: the fraction of the validation data as a float between 0 and 1.
    @return: a tuple of two tf.data.Datasets as (training, validation)
    """

    
    validation_data_percent = round(validation_data_fraction * 100)

    if not (0 <= validation_data_percent <= 100):
        raise ValueError("validation data fraction must be ∈ [0,1]")

    DATASET_SIZE = 61310 #tf.data.experimental.cardinality(dataset).numpy()
    print("Dataset Size", DATASET_SIZE)
    val_size = int(validation_data_fraction * DATASET_SIZE)
    train_size = DATASET_SIZE - val_size

    
    train_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size)


    return train_dataset, validation_dataset

def create_images_dataset():
    minibatch_size=BATCH_SIZE
    train_tfrecords=TF_RECORDS

    print ('Setting up dataset source from', train_tfrecords)

    buffer_mb   = 512
    num_threads = 2
    dset = tf.data.TFRecordDataset(train_tfrecords, compression_type='', buffer_size=buffer_mb<<20)
    # dset = dset.repeat()
    buf_size = 1000
    dset = dset.prefetch(BUFFER_SIZE)
    dataset_train, dataset_validation= split_dataset_new(dset,0.1)

    # dset = dset.shuffle(buffer_size=BUFFER_SIZE).map(parse_func).repeat().batch(minibatch_size)
    # dset = dset.map(parse_func)
    
    # dataset_train=dataset_train.repeat()
    # dataset_validation=dataset_validation.repeat()

    # dataset_train = dataset_train.batch(minibatch_size)
    # dataset_validation = dataset_validation.batch(minibatch_size)
    dataset_train=dataset_train.shuffle(buffer_size=BUFFER_SIZE).map(parse_func).batch(minibatch_size,drop_remainder=True)
    dataset_validation=dataset_validation.map(parse_func).batch(minibatch_size,drop_remainder=True)

    return dataset_train,dataset_validation;

def create_labels_dataset():
    minibatch_size=BATCH_SIZE
    train_tfrecords=TF_RECORDS

    print ('Setting up dataset source from', train_tfrecords)

    buffer_mb   = 512
    num_threads = 2
    dset = tf.data.TFRecordDataset(train_tfrecords, compression_type='', buffer_size=buffer_mb<<20)
    # dset = dset.repeat()
    buf_size = 1000
    dset = dset.prefetch(BUFFER_SIZE)
    dataset_train, dataset_validation= split_dataset_new(dset,0.1)

    # dset = dset.shuffle(buffer_size=BUFFER_SIZE).map(parse_func).repeat().batch(minibatch_size)
    # dset = dset.map(parse_func)
    
    # dataset_train=dataset_train.repeat()
    # dataset_validation=dataset_validation.repeat()

    # dataset_train = dataset_train.batch(minibatch_size)
    # dataset_validation = dataset_validation.batch(minibatch_size)
    dataset_train=dataset_train.shuffle(buffer_size=BUFFER_SIZE).map(parse_func_label).batch(minibatch_size,drop_remainder=True)
    dataset_validation=dataset_validation.map(parse_func_label).batch(minibatch_size,drop_remainder=True)

    return dataset_train, dataset_validation


def get_iterator(dataset):
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    return iterator, next_element

# def augment_rot(next_elem):
    

#     a,b=tf.random.uniform([2],maxval=4,dtype=tf.int32)

#     input=tf.image.rot90(input,a)
#     input=tf.transpose(tf.image.rot90(tf.transpose(input,[0,1,3,2]),b),[0,1,3,2])

#     target=tf.image.rot90(target,a)
#     target=tf.transpose(tf.image.rot90(tf.transpose(target,[0,1,3,2]),b),[0,1,3,2])

#     input  = tf.transpose(input, [1, 2, 3, 0])
#     target = tf.transpose(target, [1, 2, 3, 0])

if __name__ == '__main__':
    for epoch in range(3):
        print("next epoch")
        train_ds, val_ds = create_images_dataset()
        for images in train_ds.take(3):
            print(images[1].shape)
            print(tf.math.reduce_min(images[1]), tf.math.reduce_max(images[1]))
            break

