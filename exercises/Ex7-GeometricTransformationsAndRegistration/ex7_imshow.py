

import skimage
import cv2
import matplotlib.pyplot as plt
import numpy as np


data_dir = './data/'
src_img = cv2.imread(data_dir + 'Hand1.jpg')
cv2.imshow('src', src_img)