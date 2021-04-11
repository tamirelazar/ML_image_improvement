import re
import os, itertools, random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
from skimage.draw import line

# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json

from imageio import imread
from skimage.color import rgb2gray
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, UpSampling2D, Dense, Flatten, Reshape, \
    AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from skimage import color

########## Utils ##########

def relpath(path):
    """Returns the relative path to the script's location

    Arguments:
    path -- a string representation of a path.
    """
    return os.path.join(os.getcwd(), path)


def list_images(path, use_shuffle=True):
    """Returns a list of paths to images found at the specified directory.

    Arguments:
    path -- path to a directory to search for images.
    use_shuffle -- option to shuffle order of files. Uses a fixed shuffled order.
    """

    def is_image(filename):
        return os.path.splitext(filename)[-1][1:].lower() in ['jpg', 'png']

    images = list(map(lambda x: os.path.join(path, x), filter(is_image, os.listdir(path))))
    # Shuffle with a fixed seed without affecting global state
    if use_shuffle:
        s = random.getstate()
        random.seed(1234)
        random.shuffle(images)
        random.setstate(s)
    return images


def images_for_denoising():
    """Returns a list of image paths to be used for image denoising in Ex5"""
    return list_images(relpath("current/image_dataset/train"), True)


def images_for_deblurring():
    """Returns a list of image paths to be used for text deblurring in Ex5"""
    return list_images(relpath("current/text_dataset/train"), True)


def images_for_super_resolution():
    """Returns a list of image paths to be used for image super-resolution in Ex5"""
    return list_images(relpath("current/image_dataset/train"), True)


def motion_blur_kernel(kernel_size, angle):
    """Returns a 2D image kernel for motion blur effect.

    Arguments:
    kernel_size -- the height and width of the kernel. Controls strength of blur.
    angle -- angle in the range [0, np.pi) for the direction of the motion.
    """
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be an odd number!')
    if angle < 0 or angle > np.pi:
        raise ValueError('angle must be between 0 (including) and pi (not including)')
    norm_angle = 2.0 * angle / np.pi
    if norm_angle > 1:
        norm_angle = 1 - norm_angle
    half_size = kernel_size // 2
    if abs(norm_angle) == 1:
        p1 = (half_size, 0)
        p2 = (half_size, kernel_size - 1)
    else:
        alpha = np.tan(np.pi * 0.5 * norm_angle)
        if abs(norm_angle) <= 0.5:
            p1 = (2 * half_size, half_size - int(round(alpha * half_size)))
            p2 = (kernel_size - 1 - p1[0], kernel_size - 1 - p1[1])
        else:
            alpha = np.tan(np.pi * 0.5 * (1 - norm_angle))
            p1 = (half_size - int(round(alpha * half_size)), 2 * half_size)
            p2 = (kernel_size - 1 - p1[0], kernel_size - 1 - p1[1])
    rr, cc = line(p1[0], p1[1], p2[0], p2[1])
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    kernel[rr, cc] = 1.0
    kernel /= kernel.sum()
    return kernel


def read_image(filename, representation):
    """Reads an image, and if needed makes sure it is in [0,1] and in float64.
    arguments:
    filename -- the filename to load the image from.
    representation -- if 1 convert to grayscale. If 2 keep as RGB.
    """
    im = imread(filename)
    if representation == 1 and im.ndim == 3 and im.shape[2] == 3:
        im = color.rgb2gray(im).astype(np.float64)
    if im.dtype == np.uint8:
        im = im.astype(np.float64) / 255.0
    return im


########## Dataset Handling ##########

image_dict = {}

def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    A generator for generating pairs of image patches, corrupted and original
    :param filenames: a list of filenames of clean images.
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent.
    :param corruption_func: A function receiving a numpy array representation of an image as a single argument, and returning a randomly corrupted version of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract.
    :return:outputs random tuples of the form (source_batch, target_batch), where each output variable is an array of shape(batch_size, height, width, 1).
     target_batch is made of clean images and source_batch is their respective randomly corrupted version
     according to corruption_func(im)
    """
    crop_h, crop_w = crop_size
    while (True):
        orig_im_list = []
        corr_im_list = []
        rand_idx = np.random.randint(0, len(filenames), batch_size)
        for idx in rand_idx:
            filename = filenames[idx]
            if filename in image_dict.keys():
                im = image_dict[filename].copy()
            else:
                im = read_image(filename, 1)
                image_dict[filename] = im.copy()

            rows = im.shape[0]
            cols = im.shape[1]
            start_row = np.random.randint(0, rows - (crop_h * 3))
            end_row = start_row + (crop_h * 3)
            start_col = np.random.randint(0, cols - (crop_w * 3))
            end_col = start_col + (crop_w * 3)
            patch = im[start_row:end_row, start_col:end_col]

            corr = corruption_func(np.copy(patch))
            patch -= 0.5
            corr -= 0.5

            rows = patch.shape[0]
            cols = patch.shape[1]
            start_row = np.random.randint(0, rows - crop_h)
            end_row = start_row + crop_h
            start_col = np.random.randint(0, cols - crop_w)
            end_col = start_col + crop_w

            orig_im_list.append(patch[start_row:end_row, start_col:end_col, np.newaxis])
            if corr.ndim == 2:
                corr_im_list.append(corr[start_row:end_row, start_col:end_col, np.newaxis])
            else:
                corr_im_list.append(corr[start_row:end_row, start_col:end_col])

        target = np.stack(orig_im_list)
        source = np.stack(corr_im_list)
        yield (source, target)


########## Neural Network Model ##########

def resblock(input_tensor, num_channels):
    """
    Takes as input a symbolic input tensor and the number of channels for each of its convolutional layers, and returns the symbolic output tensor of the resnet block.
    The convolutional layers should use “same” border mode, so as to not decrease the spatial dimension of the output tensor.
    :param input_tensor: input tensor
    :param num_channels: number of channels
    :return: symbolic output tensor of the resnet block
    """
    x = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(num_channels, (3, 3), padding='same')(x)
    x = Add()([input_tensor, x])
    x = Activation('relu')(x)
    return x


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    Create an untrained Keras model with input dimension the shape of (height, width, 1), and all convolutional layers (including residual
    blocks) with number of output channels equal to num_channels, except the very last convolutional layer which should have a single output channel.
    The number of residual blocks should be equal to num_res_blocks.
    :param height: height
    :param width: width
    :param num_channels: number of channels
    :param num_res_blocks: number of residual blocks
    :return: an untrained Keras model.
    """
    input_ten = Input(shape=(height, width, 1))
    x = Conv2D(num_channels, (3, 3), padding='same')(input_ten)
    x = Activation('relu')(x)
    for i in range(num_res_blocks):
        x = resblock(x, num_channels)
    x = Conv2D(1, (3, 3), padding='same')(x)
    x = Add()([input_ten, x])
    model = Model(inputs=input_ten, outputs=x)
    return model


########## Training Networks for Image Restoration ##########

def split_images_badly(image_list):
    """
    Splits a list of image file paths randomly in a 80-20 ratio, but not randomly
    """
    list_length = len(image_list)
    training_idx = int(list_length * 0.8)
    training = image_list[:training_idx]
    validation = image_list[training_idx:]
    return (training, validation)


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    """
    Divide the images into a training set and validation set, using an 80-20 split, and generate from each set a dataset with the given batch size
    and corruption function. Eventually it will train the model.
    :param model:  a general neural network model for image restoration.
    :param images: a list of file paths pointing to image files. You should assume these paths are complete, and should append anything to them.
    :param corruption_func: a corruption function.
    :param batch_size: the size of the batch of examples for each iteration of SGD.
    :param steps_per_epoch: the number of update steps in each epoch.
    :param num_epochs: the number of epochs for which the optimization will run.
    :param num_valid_samples: the number of samples in the validation set to test on after every epoch.
    """
    training, validation = split_images_badly(images)
    _, h, w, _ = model.input_shape
    train_set = load_dataset(training, batch_size, corruption_func, (h, w))
    valid_set = load_dataset(validation, batch_size, corruption_func, (h, w))
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    model.fit_generator(train_set, steps_per_epoch, num_epochs, validation_data=valid_set,
                        validation_steps=num_valid_samples / batch_size, use_multiprocessing=True)


########## Image Restoration of Complete Images ##########

def restore_image(corrupted_image, base_model):
    """
    Restore full images of any size
    :param corrupted_image: a grayscale image of shape (height, width) and with values in the [0, 1] range of type float64 that is affected
    by a corruption generated from the same corruption function encountered during training (the image is not necessarily from the training set though).
    :param base_model: a neural network trained to restore small patches. The input and output of the network are images with values in the [−0.5, 0.5] range.
    :return: the restored image
    """
    h, w = corrupted_image.shape
    np.reshape(corrupted_image, [h, w, 1])
    a = Input(shape=(h, w, 1))
    b = base_model.call(a)
    new_model = Model(inputs=a, outputs=b)
    im = new_model.predict(np.copy(corrupted_image[np.newaxis, ..., np.newaxis] - 0.5))[0].astype(np.float64)
    return im[:, :, 0] + 0.5


########## Application to Image Denoising and Deblurring ##########

def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    Add random gaussian noise to an image
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian distribution.
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma, representing the maximal variance of the gaussian distribution
    :return: the corrupted image
    """
    sigma = np.random.uniform(min_sigma, max_sigma)
    noise = np.random.normal(0, sigma, image.shape)
    corr_im = np.copy(image) + noise
    corr_im *= 255
    corr_im = np.rint(corr_im)
    corr_im /= 255
    corr_im = np.clip(corr_im, 0, 1)
    return corr_im


# Needs to be set manually!
denoise_num_res_blocks = 7

def learn_denoising_model(denoise_num_res_blocks, quick_mode=False):
    """
    Train a denoising model
    :param denoise_num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """
    model = build_nn_model(24, 24, 48, denoise_num_res_blocks)
    corruption_func = lambda x: add_gaussian_noise(x, 0, 0.2)
    images = images_for_denoising()
    if quick_mode:
        train_model(model, images, corruption_func, 10, 3, 2, 30)
    else:
        train_model(model, images, corruption_func, 100, 100, 10, 1000)
    return model


# model = learn_denoising_model(denoise_num_res_blocks, False)
# image_list = images_for_denoising()
# for i in range(5):
#   image = read_image(np.random.choice(image_list), 1)
#   plt.imshow(image, cmap='gray')
#   plt.show()

#   corr = add_gaussian_noise(image, 0, 0.2)
#   plt.imshow(corr[:, :, 0], cmap='gray')
#   plt.show()

#   res_im = restore_image(corr[:, :, 0], model)[:, :, 0]
#   plt.imshow(res_im, cmap='gray')
#   plt.show()


########## Image Deblurring ##########

def add_motion_blur(image, kernel_size, angle):
    """
    Simulate motion blur on the given image using a square kernel of size kernel_size where the line has the given angle in radians, measured relative to the positive horizontal axis.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param kernel_size:  an odd integer specifying the size of the kernel.
    :param angle: an angle in radians in the range [0, π).
    :return: blurred image
    """
    kernel = motion_blur_kernel(kernel_size, angle)
    blurred = np.copy(image)
    convolve(image, kernel, mode='constant', output=blurred)
    return blurred


def random_motion_blur(image, list_of_kernel_sizes):
    """
    Simulate motion blur on the given image using a square kernel of size kernel_size where the line has the given angle in radians, measured relative to the positive horizontal axis.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param list_of_kernel_sizes: a list of odd integers.
    :return: blurred image
    """
    angle = np.random.uniform(0, np.pi)
    kernel_size = np.random.choice(list_of_kernel_sizes)
    return add_motion_blur(image, kernel_size, angle)


# Needs to be set manually!
deblur_num_res_blocks = 7

def learn_deblurring_model(deblur_num_res_blocks, quick_mode=False):
    """
    Train a deblurring model
    :param deblur_num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """
    model = build_nn_model(16, 16, 32, deblur_num_res_blocks)
    corruption_func = lambda x: random_motion_blur(x, [7])
    images = images_for_deblurring()
    if quick_mode:
        train_model(model, images, corruption_func, 10, 3, 2, 30)
    else:
        train_model(model, images, corruption_func, 100, 100, 10, 1000)
    return model


# model = learn_deblurring_model(deblur_num_res_blocks, False)
# image_list = images_for_deblurring()
# image = read_image(np.random.choice(image_list), 1)[:, :, np.newaxis]
# plt.imshow(image[:, :, 0], cmap='gray')
# plt.show()

# corr = random_motion_blur(image[:, :, 0], [7])
# plt.imshow(corr[:, :, 0], cmap='gray')
# plt.show()

# res_im = restore_image(corr[:, :, 0], model)[:, :, 0]
# plt.imshow(res_im, cmap='gray')
# plt.show()


########## Image Super-resolution ##########

def super_resolution_corruption(image, factor=None):
    """
    Perform the super resolution corruption
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :return: corrupted image
    """
    if factor is None:
        factor = np.random.randint(2, 5)
    inv_factor = 1 / factor
    width, height = image.shape[0], image.shape[1]
    image = np.copy(image[:(width // factor) * factor, :(height // factor) * factor])
    im_z = zoom(image, inv_factor)
    image = zoom(im_z, factor)
    return image


# Needs to be set manually!
super_resolution_num_res_blocks = 7
batch_size = 65
steps_per_epoch = 500
num_epochs = 8
patch_size = 32
num_channels = 64

def learn_super_resolution_model(super_resolution_num_res_blocks, quick_mode=False):
    """
    Train a super resolution model
    :param super_resolution_num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """
    model = build_nn_model(patch_size, patch_size, num_channels, super_resolution_num_res_blocks)
    corruption_func = lambda x: super_resolution_corruption(x)
    images = images_for_super_resolution()
    if quick_mode:
        train_model(model, images, corruption_func, 10, 3, 2, 30)
    else:
        train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, 1000)
    return model


# model = learn_super_resolution_model(super_resolution_num_res_blocks, True)
# for i in range(5):
#   image_list = images_for_super_resolution()
#   image = read_image(np.random.choice(image_list), 1)
#   plt.imshow(image, cmap='gray')
#   plt.show()

#   corr = super_resolution_corruption(image)
#   plt.imshow(corr[:, :, 0], cmap='gray')
#   plt.show()

#   res_im = restore_image(corr[:, :, 0], model)[:, :, 0]
#   plt.imshow(res_im, cmap='gray')
#   plt.show()
