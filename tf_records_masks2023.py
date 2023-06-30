import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob, os
import mrcfile
import time
import random

data_dirs = ['/path/to/halfmaps/']
mask_dirs = ['/path/to/masks/']
outputPath = '/path/to/your/dataset.tfrecords'
writer = tf.io.TFRecordWriter(outputPath)

def standardization(array):
    array = (array - array.mean())
    array /= np.std(array)
    # array = (array - array.min()) / (array.max() - array.min())
    # array-=0.5
    return array

def _bytes_feature(value):
    ''' 
    tf record helper function
    '''
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _write_to_tf(e, o, label, writer):
    
    ''' this takes the even and odd particles (input, target) and writes them to a tfrecords file '''

    input1 = tf.compat.as_bytes(e.tostring()) #input
    target = tf.compat.as_bytes(o.tostring())  #target 
    label  = tf.compat.as_bytes(label)


    feature = {
        'input' :_bytes_feature(input1),
        'target':_bytes_feature(target),
        'label': _bytes_feature(label)
    }

    tf_example = tf.train.Example(
        features = tf.train.Features(feature=feature))

    writer.write(tf_example.SerializeToString())   


mn=0
image_count=0
start_time = time.time()

file_list=[]


for i, data_dir in enumerate(data_dirs):
    for file_1 in glob.glob(data_dir+'/*_1.map.gz'):
        file_temp = os.path.relpath(file_1, data_dir)
        file_mask = mask_dirs[i]+file_temp[:-17]+'msk_1.map'
        
        file_list.append((file_1, file_temp, file_mask))

random.shuffle(file_list)

for files in file_list: #_1.map.gz'
    file_1, file_temp, file_mask = files
    file_2 = file_1[:-8]+'2.map.gz'

    print(file_1, file_mask)
    
    if (not os.path.isfile(file_mask)):
        file_mask = file_mask[:-6]+'.map'
        if (not os.path.isfile(file_mask)):
            continue
    
    print(file_temp)
    print(mn)
    image_1 = mrcfile.open((file_1)).data.astype(np.float32)
    image_2 = mrcfile.open(file_2).data.astype(np.float32)
    image_mask = mrcfile.open(file_mask).data.astype(np.float32)

    if (image_mask.shape!=image_1.shape):
        print("Mask Size mismatch")
        continue
    image_count + =1
    image_1 = standardization(image_1)
    image_2 = standardization(image_2)
    h,w,d = image_1.shape
    h_val = 0 #random.randint(0,h-64)

    while (h_val+96<=h):
        h_val += 96
        w_val = 0
        while (w_val+96 <= w):
            w_val += 96
            d_val = 0
            while (d_val+96 <= d):
                d_val += 96
                mask_to_write = image_mask[h_val-96:h_val, w_val-96:w_val, d_val-96:d_val]
                if (np.sum(mask_to_write) <= 0):
                    continue
                image_to_write = image_1[h_val-96:h_val, w_val-96:w_val, d_val-96:d_val] * mask_to_write
                target_to_write = image_2[h_val-96:h_val, w_val-96:w_val, d_val-96:d_val] * mask_to_write   
                mn += 1
                image_to_write = np.expand_dims(image_to_write, axis=0)
                _write_to_tf(image_to_write, target_to_write, file_temp[:-18], writer)
    
writer.close()
print("--- %s seconds ---" % (time.time() - start_time))
print("Total patches: ", mn)
print("Total maps included: ", image_count)
