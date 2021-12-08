import os
from glob import glob

import cv2
import numpy as np
from keras.preprocessing import image as kpimage


def mkdir(p):
    if not os.path.exists(p):
        os.mkdir(p)


def link(src, dst):
    if not os.path.exists(dst):
        os.symlink(src, dst)


image_files = glob("../COVID-19_Radiography_Dataset/*/*")
dataset_path = "./"


def random_apply(f, img, p=0.5):
    if np.random.rand() < p:
        return f(img)
    return img


def load_some_image(image_paths):
    return kpimage.img_to_array(kpimage.load_img(
        np.random.choice(image_paths))).astype("uint8")


def random_rotate(image, max_angle=360, min_angle=0):
    angle = np.random.rand() * (max_angle - min_angle) + min_angle
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image,
                            rot_mat,
                            image.shape[1::-1],
                            flags=cv2.INTER_LINEAR)
    return result


def random_zoom(image,
                p_zoom_in=0.5,
                max_zoom_factor=1.5,
                min_zoom_factor=0.75):
    if np.random.rand() < p_zoom_in:
        return random_zoom_in(image, max_zoom_factor)
    return random_zoom_out(image, min_zoom_factor)


def random_zoom_out(image, min_zoom_factor=0.75):
    zoom_factor = np.random.rand() * (1 - min_zoom_factor) + min_zoom_factor
    newimg = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor)
    old_size = np.array(image.shape[:-1])
    new_size = np.array(newimg.shape[:-1])
    lt = np.floor(old_size / 2 - new_size / 2).astype(int)
    rb = lt + new_size
    p = np.zeros_like(image)
    p[lt[0]:rb[0], lt[1]:rb[1]] = newimg
    return p


def random_zoom_in(image, max_zoom_factor=1.5):
    zoom_factor = 1 + np.random.rand() * (max_zoom_factor - 1)
    newimg = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor)
    new_size = np.array(newimg.shape[:-1])
    old_size = np.array(image.shape[:-1])
    lt = np.floor(new_size / 2 - old_size / 2).astype(int)
    rb = lt + old_size
    return newimg[lt[0]:rb[0], lt[1]:rb[1]]


def random_shift_h(image, max_shifted=0.2):
    d = np.random.rand() * max_shifted * 2 - max_shifted
    imgsize = image.shape[0]
    shifted = int(d * imgsize)
    p = np.zeros_like(image)
    if d > 0:
        new_size = imgsize - shifted
        p[shifted:imgsize] = image[:new_size]
    else:
        new_size = imgsize + shifted
        p[:imgsize + shifted] = image[imgsize - new_size:]
    return p


def random_shift_v(image, max_shifted=0.2):
    d = np.random.rand() * max_shifted * 2 - max_shifted
    imgsize = image.shape[1]
    shifted = int(d * imgsize)
    p = np.zeros_like(image)
    if d > 0:
        new_size = imgsize - shifted
        p[:, shifted:imgsize] = image[:, :new_size]
    else:
        new_size = imgsize + shifted
        p[:, :imgsize + shifted] = image[:, imgsize - new_size:]
    return p


def random_shift(image, max_shifted=0.2):
    return random_shift_h(random_shift_v(image, max_shifted), max_shifted)


def random_brightness(img, max_bright=0.1, min_bright=-0.1):
    value = int(
        (np.random.rand() * (max_bright - min_bright) + min_bright) * 255)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value > 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        value *= -1
        lim = value
        v[v <= lim] = 0
        v[v > lim] -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def gaus_noise(image, var=50):
    image = image.astype(int)
    row, col, ch = image.shape
    mean = 0
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, 1))
    gauss = np.clip(gauss, -255, 255).astype(int)
    gauss = gauss.reshape(row, col, 1)
    noisy = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy


def salt_pepper(image, s_vs_p=0.5, amount=0.0004):
    row, col, ch = image.shape
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [
        np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:-1]
    ]
    out[tuple(coords)] = 255

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
    coords = [
        np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:-1]
    ]
    out[tuple(coords)] = 0
    return out


def poisson_noise(image):
    im = image[:, :, :1].astype(np.float64) / 255
    vals = len(np.unique(im))
    vals = 2**np.ceil(np.log2(vals))
    noisy = np.random.poisson(im * vals) / float(vals)
    return (np.clip(np.zeros_like(image) + noisy * 255, 0,
                    255)).astype(np.uint8)


def random_aug(img):
    img = random_apply(random_rotate, img)
    p = np.random.rand()
    if p < 0.66:
        img = random_apply(random_shift_h, img, 1)
    if p > 0.33:
        img = random_apply(random_shift_v, img, 1)
    img = random_apply(random_zoom, img)
    img = random_apply(random_brightness, img, 1)
    p = np.random.rand()
    if p < 0.66:
        img = random_apply(poisson_noise, img, 1)
    if p > 0.33:
        img = random_apply(gaus_noise, img, 1)
    img = random_apply(salt_pepper, img, 0.2)

    return img
