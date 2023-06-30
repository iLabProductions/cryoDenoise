import tensorflow as tf
import os

#For distributing the code amongst multiple GPUs
mirrored_strategy = tf.distribute.MirroredStrategy()
NUM_REPLICAS=mirrored_strategy.num_replicas_in_sync
TF_RECORDS='/path/to/maskedMaps.tfrecords' # replace with path to your tfRecords file


#BATCH_SIZE_PER_REPLICA = 5
BATCH_SIZE = 6 #BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync
#LEARNING_RATES_BY_BATCH_SIZE = {2:0.001, 4:0.001, 8:0.001, 16:0.001, 32: 0.001, 64: 0.002, 128: 0.004, 256:0.006, 512:0.008}
#LEARNING_RATE = LEARNING_RATES_BY_BATCH_SIZE[BATCH_SIZE]
LEARNING_RATE = 0.0003
RAMP_DOWN_PERC = 0.3
DECAY_STEPS = 10

#Images dataset location
LOG_IMAGES_INTERVAL = 1000

# If ADD_NOISE is set to False it means that we have paired files with noise already added .
# The code will take all the filenames that match IMAGES_PATH_TRAIN and pair them with
# same filename just with changed suffix (ODD_SUFFIX -> EVEN_SUFFIX)
# The code can be found in process_image_pair function in dataset
ADD_NOISE = 'NO'
assert ADD_NOISE in ['NO', 'PERMANENT', 'EPOCH']
NOISE_TYPE = 'GAUSSIAN'
assert NOISE_TYPE in ['GAUSSIAN', 'LOGNORMAL']

#General settings
BUFFER_SIZE = 1024
if NOISE_TYPE == 'GAUSSIAN':
    VARIANCE = 0.2
else:
    VARIANCE = 1.3
LOG_PATH = '/cluster/project/cryoDenoiseLogs/' #replace with your path
EPOCHS_NO = 750

#Loss function to use
POSSIBLE_LOSSES = ['FRC', 'L2', 'L1', 'W_L2','W_FRC','CREF_L1']
LOSS_FUNCTION = 'CREF_L1'
USE_BIAS = False
SAVED_MODEL_LOGDIR = None
RESTORE_EPOCH = 0
assert LOSS_FUNCTION in POSSIBLE_LOSSES
EPOCH_FILEPATTERN = "saved-model-epoch-{}"
BATCH_FILEPATTERN = "saved-model-batch-{}"




try:
    from local_settings import *
except ImportError:
    pass
BATCHES_NUMBER = int(55179//BATCH_SIZE) #replace with your dataset size

if ADD_NOISE == 'NO':
    # BATCHES_NUMBER=int(50000/BATCH_SIZE) #int(BATCHES_NUMBER/2) + 1
    # BATCHES_NUMBER=int(BATCHES_NUMBER/2) + 1
    pass
