import random
import nibabel as nib
import tensorflow as tf
from scipy import ndimage


def read_file(file_path):
    """ Read and load volume """
    scan = nib.load(filename=file_path)
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """ Normalize the volume """
    minimum = -1000
    maximum = 400

    volume[volume < minimum] = minimum
    volume[volume > maximum] = maximum

    volume = (volume - minimum) / (maximum - minimum)
    volume = volume.astype('float32')

    return volume


def resize_volume(img):
    """ Resize volume across z-axis """
    desired_depth = 64
    desired_width = 128
    desired_height = 128

    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]

    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height

    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    img = ndimage.rotate(input=img, angle=90, reshape=False)
    img = ndimage.zoom(input=img, zoom=(width_factor, height_factor, depth_factor), order=1)

    return img


def process_scan(path):
    """ Process volume """
    volume = read_file(file_path=path)
    volume = normalize(volume=volume)
    volume = resize_volume(img=volume)

    return volume


def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume_):
        angles = [-20, -10, -5, 5, 10, 20]
        angle = random.choice(angles)
        volume_ = ndimage.rotate(input=volume_, angle=angle, reshape=False)
        volume_[volume_ < 0] = 0
        volume_[volume_ > 1] = 1

        return volume_

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)

    return augmented_volume


def train_prepare(volume, label):
    volume = rotate(volume=volume)
    volume = tf.expand_dims(input=volume, axis=3)

    return volume, label


def validation_prepare(volume, label):
    volume = rotate(volume=volume)
    volume = tf.expand_dims(input=volume, axis=3)

    return volume, label
