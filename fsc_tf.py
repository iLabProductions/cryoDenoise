import tensorflow as tf
import numpy as np

class loss_fn:
    def __init__(self, *args, **kwargs):
        self.radial_masks, self.spatial_freq = self.get_radial_masks()
        
    def compute_loss_frc(self, img1, img2):
        denoised1 = img1
        loss_img = self.fourier_ring_correlation(img2, denoised1, self.radial_masks, self.spatial_freq)
        return loss_img



    @tf.function
    def fourier_ring_correlation(self, image1, image2, rn, spatial_freq):
        # we need the channels first format for this loss
        image1 = tf.transpose(image1, perm=[0, 4, 1, 2, 3])
        image2 = tf.transpose(image2, perm=[0, 4, 1, 2, 3])
        image1 = tf.compat.v1.cast(image1, tf.complex64)
        image2 = tf.compat.v1.cast(image2, tf.complex64)
        fft_image1 = tf.signal.fftshift(tf.signal.fft3d(image1), axes=[2, 3, 4])
        fft_image2 = tf.signal.fftshift(tf.signal.fft3d(image2), axes=[2, 3, 4])
        rn=tf.cast(rn, tf.complex64)
        t1 = tf.multiply(fft_image1, rn)  # (128, BS?, 3, 256, 256)
        t2 = tf.multiply(fft_image2, rn)
        c1 = tf.math.real(tf.reduce_sum(tf.multiply(t1, tf.math.conj(t2)), [2, 3, 4, 5]))

        
        c2 = tf.reduce_sum(tf.math.abs(t1) ** 2, [2, 3, 4, 5])
        c3 = tf.reduce_sum(tf.math.abs(t2) ** 2, [2, 3, 4, 5])
        zero_temp=tf.zeros([1], tf.float32)
        norm_prod=tf.math.sqrt(tf.math.multiply(c2, c3))
        frc = tf.where(tf.math.equal(norm_prod, tf.zeros_like(norm_prod)),zero_temp,tf.math.divide(c1,tf.where(tf.math.equal(norm_prod, tf.zeros_like(norm_prod)),tf.ones_like(norm_prod),norm_prod)))
        frc = tf.where(tf.compat.v1.is_inf(frc), tf.zeros_like(frc), frc)  # inf
        
        t = spatial_freq
        y = frc

        return t.numpy(),y.numpy()


    @staticmethod
    def radial_mask(r, cx=48, cy=48, cz=48, sx=np.arange(0, 96), sy=np.arange(0, 96), sz=np.arange(0, 96), delta=1):

        x2, x1, x0 = np.meshgrid(sx-cx,sy-cy,sz-cz, indexing='ij')

        coords = np.stack((x0,x1,x2), -1)

        ind = (coords**2).sum(-1)
        ind1 = ind <= ((r[0] + delta) ** 2)  # one liner for this and below?
        ind2 = ind > (r[0] ** 2)
        return ind1 * ind2


    def get_radial_masks(self):
        freq_nyq = int(np.floor(int(96) / 2.0))
        radii = np.arange(48).reshape(48, 1)  # image size 96, binning = 1
        radial_masks = np.apply_along_axis(self.radial_mask, 1, radii, 48, 48, 48, np.arange(0, 96), np.arange(0, 96), np.arange(0, 96), 5)
        radial_masks = np.expand_dims(radial_masks, 1)
        radial_masks = np.expand_dims(radial_masks, 1)
        spatial_freq = radii.astype(np.float32) / freq_nyq
        spatial_freq = spatial_freq / max(spatial_freq)

        return radial_masks, spatial_freq