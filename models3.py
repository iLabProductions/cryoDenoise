from functools import partial
import json
import os
import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
import glob

from dataset import create_images_dataset
import settings as settings
from settings import (LOSS_FUNCTION, USE_BIAS, EPOCHS_NO, LEARNING_RATE, BATCH_SIZE,
                      ADD_NOISE, BATCHES_NUMBER, RAMP_DOWN_PERC, DECAY_STEPS,
                      SAVED_MODEL_LOGDIR, RESTORE_EPOCH, EPOCH_FILEPATTERN, BATCH_FILEPATTERN, NUM_REPLICAS)
from utils import get_new_model_log_path

from network import autoencoder_3D

	
class FRCUnetModel(tf.keras.Model):
    def __init__(self, logdir, model_path=None,  *args, **kwargs):
        super(FRCUnetModel, self).__init__(**kwargs)
        self.model = autoencoder_3D(*args, **kwargs)
        if model_path is not None:
            self.model.load_weights(model_path)
        self.radial_masks, self.spatial_freq = self.get_radial_masks()
        if logdir is not None:
            self.writer = tf.summary.create_file_writer(logdir)


    def call(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def scale(image):
        scaled = image - tf.math.reduce_min(image)
        return scaled / tf.math.reduce_max(scaled)

    @staticmethod
    def scale_back(image,original):
        scaled = image - tf.math.reduce_mean(image)
        scaled/=tf.math.reduce_std(scaled)
        scaled=scaled*tf.math.reduce_std(original)+tf.math.reduce_mean(original)
        return scaled

    def create_image_summaries(self, images_data, denoised, step, mode='train'):
        original, noisy1, noisy2 = images_data

        tf.summary.image("original_" + mode, self.scale(original[...,48,:]), step=step)
        tf.summary.image("noisy1_" + mode, self.scale(noisy1[...,48,:]), step=step)
        tf.summary.image("noisy2_" + mode, self.scale(noisy2[...,48,:]), step=step)

        tf.summary.image("denoised_ " + mode, self.scale(denoised[...,48,:]), step=step)

    @tf.function
    def compute_loss_mae(self, data):
        original, img1, img2 = data
        denoised = self.call(img1)
        loss = tf.math.reduce_mean(tf.keras.losses.MAE(denoised, img2))

        loss_original = tf.math.reduce_mean(tf.keras.losses.MSE(denoised, original))
        return denoised, 0, 0, loss, loss_original, 0

    @tf.function
    def compute_loss_mse(self, data):
        original, img1, img2 = data
        denoised = self.call(img1)
        loss = tf.math.reduce_mean(tf.keras.losses.MSE(denoised, img2))

        loss_original = tf.math.reduce_mean(tf.keras.losses.MSE(denoised, original))
        return denoised, 0, 0, loss, loss_original, 0

    @tf.function
    def compute_loss_weighed_mse(self, data):
        original, img1, img2 = data
        denoised = self.call(img1)
        loss_scale=tf.squeeze(0.5*tf.math.abs(img1+img2),axis=-1)
        loss_scale/=tf.math.reduce_sum(loss_scale)

        loss = tf.math.reduce_sum(loss_scale*tf.keras.losses.MSE(denoised, img2))

        loss_original = tf.math.reduce_mean(tf.keras.losses.MSE(denoised, original))
        loss_img=tf.math.reduce_mean(self.fourier_ring_correlation(img2, denoised, self.radial_masks, self.spatial_freq))
        return denoised, loss_img, 0, loss, loss_original, 0

    @tf.function
    def compute_loss_frc(self, data):
        original, img1, img2 = data
        denoised1 = self.call(img1)
        
        loss_img = 1-self.fourier_ring_correlation(img2, denoised1, self.radial_masks, self.spatial_freq)
        if USE_BIAS:
            # source: https://www.biorxiv.org/content/10.1101/2020.08.16.253070v1.full.pdf
            # eq.18 in supplementary
            # var(B) = cov(D1,D2) + [cov(M1,M1)cov(M2,M2)]½ - 2cov(M1,D2)
            #cov_m1_m2 = self.fourier_corr(img1, img2, self.radial_masks,
            #                                                        self.spatial_freq)
            #cov_d1_d2 = self.covariance(denoised1, denoised2)

            #cov_m1_d2 = self.covariance(img1, denoised2)
            #loss_bias = (cov_d1_d2 - 2 * cov_m1_d2)# + cov_m1_m2
            #loss_cov_d1_d2 = tf.math.reduce_mean(cov_d1_d2)
            denoised2 = self.call(img2)

            loss_bias = tf.math.reduce_mean(self.fourier_ring_correlation(denoised1, denoised2, self.radial_masks,
                                                      self.spatial_freq))
            loss_cov_d1_d2 = 0
            factor=4
        else:
            loss_cov_d1_d2 = 0
            loss_bias = 0
            factor=1
        loss = factor*loss_img + loss_bias
        loss = tf.math.reduce_mean(loss)
        loss_original =tf.math.reduce_mean(tf.keras.losses.MSE(self.scale_back(denoised1,original), original))
        # tf.math.reduce_mean(tf.keras.losses.MSE((denoised1-tf.math.reduce_mean(denoised1))/tf.math.reduce_std(denoised1)*tf.math.reduce_std(original)+tf.math.reduce_mean(original), original))
        loss_img = tf.math.reduce_mean(loss_img)
        loss_bias = tf.math.reduce_mean(loss_bias)
        return denoised1, loss_img, loss_bias, loss, loss_original, loss_cov_d1_d2


    @tf.function
    def compute_loss_weighed_frc(self, data):
        original, img1, img2 = data
        denoised1 = self.call(img1)
        # denoised1=self.scale_back(denoised1,img1)

        loss_img = 1-self.fourier_ring_correlation(img2, denoised1, self.radial_masks, self.spatial_freq)
        if USE_BIAS:

            denoised2 = self.call(img2)

            loss_bias = tf.math.reduce_mean(self.fourier_ring_correlation(denoised1, denoised2, self.radial_masks,
                                                      self.spatial_freq))
            loss_cov_d1_d2 = 0
            factor=4
        else:
            loss_cov_d1_d2 = 0
            loss_bias = 0
            factor=1

        loss_scale=tf.math.reduce_mean(tf.squeeze(0.5*tf.math.abs(img1+img2)))
        loss = factor*loss_img + loss_bias
        loss*=loss_scale
        loss=tf.math.reduce_mean(loss)
        loss_original =tf.math.reduce_mean(tf.keras.losses.MSE(self.scale_back(denoised1,original), original))
        # tf.math.reduce_mean(tf.keras.losses.MSE((denoised1-tf.math.reduce_mean(denoised1))/tf.math.reduce_std(denoised1)*tf.math.reduce_std(original)+tf.math.reduce_mean(original), original))
        loss_img = tf.math.reduce_mean(loss_img)
        loss_bias = tf.math.reduce_mean(loss_bias)
        return denoised1, loss_img, loss_bias, loss, loss_original, loss_cov_d1_d2

    @tf.function
    def compute_loss_l1_cref(self, data):
        original, img1, img2 = data
        noisy=(img1+img2)/2
        denoised = self.call(noisy)
        fsc_noisy= tf.abs(self.fourier_ring_correlation_unsummed(img1, img2, self.radial_masks))
        fsc_denoised=tf.abs(self.fourier_ring_correlation_unsummed(denoised, noisy, self.radial_masks))
        cref=(tf.math.sqrt(tf.math.divide_no_nan(tf.math.multiply(2.0,fsc_noisy), 1.0+fsc_noisy)))

        y=tf.abs(fsc_denoised-cref)
        t= self.spatial_freq

        loss_img = tf.reduce_sum(tf.multiply(t[1:] - t[:-1], (y[:-1] + y[1:]) / 2.), 0) #Cref Loss
        loss_img = tf.math.reduce_mean(loss_img)

        y2=0
        loss_cov_d1_d2= tf.math.reduce_mean(y2)
        loss_original =tf.math.reduce_mean(tf.keras.losses.MSE(self.scale_back(denoised,noisy), noisy))

        loss = loss_img
        loss_bias = 0
        return denoised, loss_img, loss_bias, loss, loss_original, loss_cov_d1_d2


    # @tf.function
    def infer(self, data):
        if LOSS_FUNCTION == 'FRC':
            return self.compute_loss_frc(data)
        elif LOSS_FUNCTION == 'L2':
            return self.compute_loss_mse(data)
        elif LOSS_FUNCTION == 'L1':
            return self.compute_loss_mae(data)
        elif LOSS_FUNCTION == 'W_L2':
            return self.compute_loss_weighed_mse(data)
        elif LOSS_FUNCTION == 'W_FRC':
            return self.compute_loss_weighed_frc(data)
        elif LOSS_FUNCTION == 'CREF_L1':
            return self.compute_loss_l1_cref(data)

    # @tf.function
    def train_step(self, data):
        #data contains original, img1, img2.
        with tf.GradientTape() as tape:
            denoised, loss_img, loss_bias, loss, loss_original, loss_cov_d1_d2 = self.infer(data)
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads_vars = zip(grads, self.model.trainable_variables)
        self.optimizer.apply_gradients(grads_vars)
        return {'loss': loss, 'loss_img': loss_img, 'loss_cov_d1_d2': loss_cov_d1_d2,'loss_original': loss_original}

    # @tf.function
    # def distributed_train_step(dataset_inputs):
    # per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    # return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
    #                        axis=None)

    @tf.function
    def test_step(self, data):
        denoised, loss_img, loss_bias, loss, loss_original, loss_cov_d1_d2 = self.infer(data)
        return {'loss': loss, 'loss_img': loss_img, 'loss_cov_d1_d2': loss_cov_d1_d2,'loss_original': loss_original}

    @tf.function
    def covariance(self, x, y):
        xbar = tf.reduce_mean(x, axis=[1, 2, 3])
        xbar = tf.transpose(tf.ones([256, 256, 1]) * xbar, [2, 0, 1])
        xbar = tf.reshape(xbar, [BATCH_SIZE, 256, 256, 1])
        ybar = tf.reduce_mean(y, axis=[1, 2, 3])
        ybar = tf.transpose(tf.ones([256, 256, 1]) * ybar, [2, 0, 1])
        ybar = tf.reshape(ybar, [BATCH_SIZE, 256, 256, 1])

        multiplied = (x - xbar) * (y - ybar)
        s = tf.reduce_sum(multiplied, axis=[1, 2, 3])
        return s / tf.cast(tf.size(x[0]) - 1, tf.float32)

    # @tf.function
    # def fourier_corr(self, image1, image2, rn, spatial_freq):
    #     image1 = tf.transpose(image1, perm=[0, 3, 1, 2])
    #     image2 = tf.transpose(image2, perm=[0, 3, 1, 2])
    #     image1 = tf.cast(image1, tf.complex64)
    #     image2 = tf.cast(image2, tf.complex64)
    #     rn = tf.cast(rn, tf.complex64)
    #     fft_image1 = tf.signal.fftshift(tf.signal.fft2d(image1), axes=[2, 3])
    #     fft_image2 = tf.signal.fftshift(tf.signal.fft2d(image2), axes=[2, 3])

    #     t1 = tf.multiply(fft_image1, rn)  # (128, BS?, 3, 256, 256)
    #     t2 = tf.multiply(fft_image2, rn)
    #     c1 = tf.math.real(tf.reduce_sum(tf.multiply(t1, tf.math.conj(t2)), [0, 2, 3, 4]))
    #     return c1

    @tf.function
    def fourier_ring_correlation_unsummed(self, image1, image2, rn):
        # we need the channels first format for this loss
        image1 = tf.transpose(image1, perm=[0, 4, 1, 2, 3])
        image2 = tf.transpose(image2, perm=[0, 4, 1, 2, 3])
        image1 = tf.compat.v1.cast(image1, tf.complex64)
        image2 = tf.compat.v1.cast(image2, tf.complex64)
        print('img ', image1.shape)
        fft_image1 = tf.signal.fftshift(tf.signal.fft3d(image1), axes=[2, 3, 4])
        fft_image2 = tf.signal.fftshift(tf.signal.fft3d(image2), axes=[2, 3, 4])
        rn=tf.cast(rn, tf.complex64)
        t1 = tf.multiply(fft_image1, rn)  # (128, BS?, 3, 256, 256)
        t2 = tf.multiply(fft_image2, rn)
        c1 = tf.math.real(tf.reduce_sum(tf.multiply(t1, tf.math.conj(t2)), [2, 3, 4, 5]))
        print('t1 ', t1.shape)
        print('c1 ', c1.shape)
        
        c2 = tf.reduce_sum(tf.math.abs(t1) ** 2, [2, 3, 4, 5])
        c3 = tf.reduce_sum(tf.math.abs(t2) ** 2, [2, 3, 4, 5])
        zero_temp=tf.zeros([1], tf.float32)
        norm_prod=tf.math.sqrt(tf.math.multiply(c2, c3))
        frc = tf.where(tf.math.equal(norm_prod, tf.zeros_like(norm_prod)),zero_temp,tf.math.divide(c1,tf.where(tf.math.equal(norm_prod, tf.zeros_like(norm_prod)),tf.ones_like(norm_prod),norm_prod)))
        frc = tf.where(tf.compat.v1.is_inf(frc), tf.zeros_like(frc), frc)  # inf
        frc = tf.where(tf.compat.v1.is_nan(frc), tf.zeros_like(frc), frc)  # nan
        print('frc', frc.shape)
        
        y = frc
        print('y', y.shape)
        return y

    @tf.function
    def fourier_ring_correlation(self, image1, image2, rn, spatial_freq):
        # we need the channels first format for this loss
        image1 = tf.transpose(image1, perm=[0, 4, 1, 2, 3])
        image2 = tf.transpose(image2, perm=[0, 4, 1, 2, 3])
        image1 = tf.compat.v1.cast(image1, tf.complex64)
        image2 = tf.compat.v1.cast(image2, tf.complex64)
        print('img ', image1.shape)
        fft_image1 = tf.signal.fftshift(tf.signal.fft3d(image1), axes=[2, 3, 4])
        fft_image2 = tf.signal.fftshift(tf.signal.fft3d(image2), axes=[2, 3, 4])
        rn=tf.cast(rn, tf.complex64)
        t1 = tf.multiply(fft_image1, rn)  # (128, BS?, 3, 256, 256)
        t2 = tf.multiply(fft_image2, rn)
        c1 = tf.math.real(tf.reduce_sum(tf.multiply(t1, tf.math.conj(t2)), [2, 3, 4, 5]))
        print('t1 ', t1.shape)
        print('c1 ', c1.shape)
        
        c2 = tf.reduce_sum(tf.math.abs(t1) ** 2, [2, 3, 4, 5])
        c3 = tf.reduce_sum(tf.math.abs(t2) ** 2, [2, 3, 4, 5])
        zero_temp=tf.zeros([1], tf.float32)
        norm_prod=tf.math.sqrt(tf.math.multiply(c2, c3))
        frc = tf.where(tf.math.equal(norm_prod, tf.zeros_like(norm_prod)),zero_temp,tf.math.divide(c1,tf.where(tf.math.equal(norm_prod, tf.zeros_like(norm_prod)),tf.ones_like(norm_prod),norm_prod)))
        frc = tf.where(tf.compat.v1.is_inf(frc), tf.zeros_like(frc), frc)  # inf
        # frc = tf.where(tf.compat.v1.is_nan(frc), tf.zeros_like(frc), frc)  # nan
        print('frc', frc.shape)
        
        t = spatial_freq
        y = frc
        print('y', y.shape)
        
        riemann_sum = tf.reduce_sum(tf.multiply(t[1:] - t[:-1], (y[:-1] + y[1:]) / 2.), 0)
        return riemann_sum

    @tf.function
    def power_spectra_covariance(self, image1,image2, rn):
        # we need the channels first format for this loss
        image1 = tf.transpose(image1, perm=[0, 4, 1, 2, 3])
        image1 = tf.compat.v1.cast(image1, tf.complex64)
        image2 = tf.transpose(image2, perm=[0, 4, 1, 2, 3])
        image2 = tf.compat.v1.cast(image2, tf.complex64)
        fft_image1 = tf.signal.fftshift(tf.signal.fft3d(image1), axes=[2, 3, 4])
        fft_image2 = tf.signal.fftshift(tf.signal.fft3d(image2), axes=[2, 3, 4])
        rn=tf.cast(rn, tf.complex64)
        t1 = tf.multiply(fft_image1, rn)  # (128, BS?, 3, 256, 256)
        t2 = tf.multiply(fft_image2, rn)

        c1 = tf.math.real(tf.reduce_mean(tf.multiply(t1, tf.math.conj(t2)), [2, 3, 4, 5]))
        c1=tf.where(tf.compat.v1.is_nan(c1), tf.zeros_like(c1), c1)  
        y=c1

        return y

    @tf.function
    def power_spectra_variance(self, image1, rn):
        # we need the channels first format for this loss
        image1 = tf.transpose(image1, perm=[0, 4, 1, 2, 3])
        image1 = tf.compat.v1.cast(image1, tf.complex64)
        fft_image1 = tf.signal.fftshift(tf.signal.fft3d(image1), axes=[2, 3, 4])
        rn=tf.cast(rn, tf.complex64)
        t1 = tf.multiply(fft_image1, rn)  # (128, BS?, 3, 256, 256)

        c1 = tf.math.real(tf.reduce_mean(tf.multiply(t1, tf.math.conj(t1)), [2, 3, 4, 5]))
        c1=tf.where(tf.compat.v1.is_nan(c1), tf.zeros_like(c1), c1)  
        y=c1

        return y


    def radial_mask(self, r, cx=48, cy=48, cz=48, sx=np.arange(0, 96), sy=np.arange(0, 96), sz=np.arange(0, 96), delta=1):

        x2, x1, x0 = np.meshgrid(sx-cx,sy-cy,sz-cz, indexing='ij')

        coords = np.stack((x0,x1,x2), -1)

        ind = (coords**2).sum(-1)
        ind1 = ind <= ((r[0] + delta) ** 2)  # one liner for this and below?
        ind2 = ind > (r[0] ** 2)
        return ind1 * ind2


    
    @tf.function
    def get_radial_masks(self):
        freq_nyq = int(np.floor(int(96) / 2.0))
        radii = np.arange(48).reshape(48, 1)  # image size 96, binning = 1
        radial_masks = np.apply_along_axis(self.radial_mask, 1, radii, 48, 48, 48, np.arange(0, 96), np.arange(0, 96), np.arange(0, 96), 1)
        radial_masks = np.expand_dims(radial_masks, 1)
        radial_masks = np.expand_dims(radial_masks, 1)
        spatial_freq = radii.astype(np.float32) / freq_nyq
        spatial_freq = spatial_freq / max(spatial_freq)

        return radial_masks, spatial_freq


class Summaries(tf.keras.callbacks.Callback):
    def __init__(self, epoch_restored=-1):
        self.batch_no = 0
        self.epoch_restored = epoch_restored
        print('#CALLBACK# START')
        self.train_ds, self.val_ds = create_images_dataset()

    def on_train_begin(self,logs=None):
        self.dist_train_ds = self.model.distribute_strategy.experimental_distribute_dataset(self.train_ds.take(100).repeat())
        self.dist_val_ds = self.model.distribute_strategy.experimental_distribute_dataset(self.val_ds.take(100).repeat())
        self.iterator_train = iter(self.dist_train_ds)
        self.iterator_val = iter(self.dist_val_ds)

    def on_epoch_end(self, epoch, logs):
        # print('#CALLBACK#EPOCH')
        ##Figure out how to call model in init() and distribute there
        # self.dist_train_ds = self.model.distribute_strategy.experimental_distribute_dataset(self.train_ds)
        # self.dist_val_ds = self.model.distribute_strategy.experimental_distribute_dataset(self.val_ds)
        current_epoch = epoch + self.epoch_restored + 1
        overall_loss_train, overall_loss_val = 0, 0
        overall_loss_original_train, overall_loss_original_val = 0, 0 
        overall_loss_img_train, overall_loss_img_val = 0, 0
        overall_loss_bias_train, overall_loss_bias_val = 0, 0
        overall_loss_cov_d1_d2_train, overall_loss_cov_d1_d2_val = 0, 0
        overall_batches_train, overall_batches_val = 0, 0
        # for train_images in self.train_ds.take(100):
        # iterator_train = iter(self.dist_train_ds.take(100).repeat())
        # iterator_val = iter(self.dist_val_ds.take(100).repeat())
        for i in range(100):
            train_images_dist=next(self.iterator_train)
            per_replica_result =self.model.distribute_strategy.run(self.model.infer,args=(train_images_dist,))
            results=[]
            for idx,items in enumerate(per_replica_result):
                if idx==0:
                    results.append(items.values[0])
                else:
                    results.append(sum(items.values))

            train_denoised, loss_img, loss_bias, loss_train, loss_original, loss_cov_d1_d2=results
            overall_loss_train += loss_train
            overall_loss_original_train += loss_original
            overall_loss_img_train += loss_img
            overall_loss_bias_train += loss_bias
            overall_loss_cov_d1_d2_train += loss_cov_d1_d2
            overall_batches_train += 1
        for i in range(100):
            val_images_dist=next(self.iterator_val)
            per_replica_result =self.model.distribute_strategy.run(self.model.infer,args=(val_images_dist,))
            results_val=[]
            for idx,items in enumerate(per_replica_result):
                if idx==0:
                    results_val.append(items.values[0])
                else:
                    results_val.append(sum(items.values))
            val_denoised, loss_img, loss_bias, loss_val, loss_original, loss_cov_d1_d2 = results_val           
            overall_loss_val += loss_val
            overall_loss_original_val += loss_original
            overall_loss_img_val += loss_img
            overall_loss_bias_val += loss_bias
            overall_loss_cov_d1_d2_val += loss_cov_d1_d2
            overall_batches_val += 1

        train_images=(train_images_dist[0].values[0],train_images_dist[1].values[0],train_images_dist[2].values[0])
        val_images=(val_images_dist[0].values[0],val_images_dist[1].values[0],val_images_dist[2].values[0])

        with self.model.writer.as_default():
            tf.summary.scalar("loss_epoch_train", overall_loss_train/overall_batches_train, step=current_epoch)
            tf.summary.scalar("loss_epoch_val", overall_loss_val/overall_batches_val, step=current_epoch)
            tf.summary.scalar("loss_epoch_original_train", overall_loss_original_train/overall_batches_train, step=current_epoch)
            tf.summary.scalar("loss_epoch_original_val", overall_loss_original_val/overall_batches_val,step=current_epoch)
            tf.summary.scalar("loss_epoch_img_train", overall_loss_img_train/overall_batches_train, step=current_epoch)
            tf.summary.scalar("loss_epoch_img_val", overall_loss_img_val/overall_batches_val, step=current_epoch)
            tf.summary.scalar("loss_epoch_bias_train", overall_loss_bias_train/overall_batches_train, step=current_epoch)
            tf.summary.scalar("loss_epoch_bias_val", overall_loss_bias_val/overall_batches_val, step=current_epoch)
            tf.summary.scalar("loss_epoch_cov_d1_d2_train", overall_loss_cov_d1_d2_train/overall_batches_train, step=current_epoch)
            tf.summary.scalar("loss_epoch_cov_d1_d2_val", overall_loss_cov_d1_d2_val/overall_batches_val, step=current_epoch)

            self.model.create_image_summaries(train_images, train_denoised,
                                              current_epoch, mode='train')
            self.model.create_image_summaries(val_images, val_denoised,
                                              current_epoch, mode='val')

    def on_train_end(self, logs=None):
        self.dist_train_ds = self.model.distribute_strategy.experimental_distribute_dataset(self.train_ds)
        self.dist_val_ds = self.model.distribute_strategy.experimental_distribute_dataset(self.val_ds)
        overall_loss_train, overall_loss_val = 0, 0
        overall_loss_original_train, overall_loss_original_val = 0, 0 
        overall_loss_img_train, overall_loss_img_val = 0, 0
        overall_loss_bias_train, overall_loss_bias_val = 0, 0
        overall_loss_cov_d1_d2_train, overall_loss_cov_d1_d2_val = 0, 0
        overall_batches_train, overall_batches_val = 0, 0
        iterator_train = iter(self.dist_train_ds)
        iterator_val = iter(self.dist_val_ds)
        for train_images_dist in iterator_train:
            per_replica_result =self.model.distribute_strategy.run(self.model.infer,args=(train_images_dist,))
            results=[]
            for idx,items in enumerate(per_replica_result):
                if idx==0:
                    results.append(items.values[0])
                else:
                    results.append(tf.math.reduce_mean(items.values))

            _, loss_img, loss_bias, loss_train, loss_original, loss_cov_d1_d2=results

            overall_loss_train += loss_train
            # overall_loss_original_train += loss_original
            # overall_loss_img_train += loss_img
            # overall_loss_bias_train += loss_bias
            # overall_loss_cov_d1_d2_train += loss_cov_d1_d2
            overall_batches_train += 1
        for val_images_dist in iterator_val:
            per_replica_result =self.model.distribute_strategy.run(self.model.infer,args=(val_images_dist,))
            results_val=[]
            for idx,items in enumerate(per_replica_result):
                if idx==0:
                    results_val.append(items.values[0])
                else:
                    results_val.append(np.mean(items.values))
            _, loss_img, loss_bias, loss_val, loss_original, loss_cov_d1_d2 = results_val
            overall_loss_original_val += loss_original
            overall_loss_img_val += loss_img
            overall_loss_bias_val += loss_bias
            overall_loss_cov_d1_d2_val += loss_cov_d1_d2
            overall_batches_val += 1

        with self.model.writer.as_default():
            tf.summary.scalar("loss_final_train", overall_loss_train/overall_batches_train, step=0)
            tf.summary.scalar("loss_epoch_val", overall_loss_val/overall_batches_val, step=0)


class SaveModel(tf.keras.callbacks.Callback):
    def __init__(self, logdir, epoch_restored=-1):
        self.logdir = logdir
        self.epoch_restored = epoch_restored
        self.epoch = self.epoch_restored + 1

    def on_epoch_end(self, epoch, logs):
        filename = self.logdir + '/' + EPOCH_FILEPATTERN.format(self.epoch)
        self.model.model.save_weights(filename)
        self.epoch = epoch + 1 + self.epoch_restored + 1


    def on_batch_end(self, batch, logs):
        #the batch arg i the batch in a certain epoch, and we need batch overall
        batch_no = self.epoch * BATCHES_NUMBER + batch
        #check if batch number is a power of two. We will need those model dumps for charts.
        if (batch_no & (batch_no-1) == 0):
            filename = self.logdir + '/' + BATCH_FILEPATTERN.format(batch_no)
            model.model.save_weights(filename)


def exponential_decay(epoch_no):
    return LEARNING_RATE * (1-RAMP_DOWN_PERC) ** (epoch_no/DECAY_STEPS)

class LearningRateSchedulerWithLogs(tf.keras.callbacks.LearningRateScheduler):
    def on_epoch_begin(self, epoch, logs=None):
        super(LearningRateSchedulerWithLogs, self).on_epoch_begin(epoch, logs)
        with self.model.writer.as_default():
            tf.summary.scalar("learning_rate", self.model.optimizer.lr, step=epoch)

# @tf.function
# def create_model(logdir, model_path):
# 	return FRCUnetModel(logdir, model_path)

if __name__ == "__main__":
    # tf.random.set_seed(543)

    PARAMS={k: v for k, v in vars(settings).items() if k.isupper()}



    if SAVED_MODEL_LOGDIR:
        logdir = SAVED_MODEL_LOGDIR
        model_path = SAVED_MODEL_LOGDIR + '/' + EPOCH_FILEPATTERN.format(RESTORE_EPOCH)
        save_model_callback = SaveModel(logdir, RESTORE_EPOCH)
        summaries_callback = Summaries(RESTORE_EPOCH)
    else:
        _, _, logdir = get_new_model_log_path()
        os.makedirs(logdir)
        with open(os.path.join(logdir, 'params.json'), 'w') as f:
            f.write(json.dumps(PARAMS))
        model_path = None
        save_model_callback = SaveModel(logdir)
        summaries_callback = Summaries()

    mirrored_strategy = tf.distribute.MirroredStrategy()
    
    with mirrored_strategy.scope():
	    model = FRCUnetModel(logdir, model_path=model_path)
	    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
	    model.compile(optimizer)

	    callbacks = [
        save_model_callback,
        tf.keras.callbacks.TensorBoard(logdir + '/logs', update_freq='batch', profile_batch='10,20'),
        summaries_callback,
        LearningRateSchedulerWithLogs(exponential_decay)
       ]
    train_ds, val_ds = create_images_dataset()
    model.fit(train_ds, callbacks=callbacks, epochs=EPOCHS_NO, validation_data=val_ds, validation_steps=20)
    model.model.save_weights(logdir + '/final_model')

