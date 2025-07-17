import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

import img_util

DEFAULT_IMG_DIR = 'RTImg/'


def load_images(folder=DEFAULT_IMG_DIR):
    images = []
    distances = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = img_util.readImageBW(path)
        if img is not None:
            images.append(img)
            try:
                distances.append(int(file[9:13]))
            except ValueError:
                print(f"Warning: Could not parse distance from filename: {file}")
                distances.append(0)
            print(file)
    return images, distances


def get_sharpness(images):
    sharpness_vals = []
    derivative_imgs = []
    for img in images:
        derivative = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        derivative_imgs.append(derivative)
        sharpness_vals.append(cv2.norm(derivative))
    return derivative_imgs, sharpness_vals


def get_points(subject_name='111', show_images=False):
    images, distances = load_images(folder=os.path.join(DEFAULT_IMG_DIR, subject_name))
    derivative_imgs, sharpness_vals = get_sharpness(images)

    for i, img in enumerate(derivative_imgs):
        if show_images:
            img_util.plot(img, f'IMAGE {i}', cmap='gray')
        img_util.saveImageBW(f'img_{i}.png', img)
        print(f'Sharpness for img {i}: {sharpness_vals[i]:.4f}')

    return sorted(zip(distances, sharpness_vals))


if __name__ == '__main__':
    subject = sys.argv[1] if len(sys.argv) >= 2 else '001'
    points = get_points(subject_name=subject, show_images=False)
    print(points)
    img_util.plotPoints(points, title="Sharpness vs Focus Distance")
