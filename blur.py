from sklearn.cluster import KMeans

import argparse
import cv2
import numpy as np
import utils

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image to blur")

args = vars(ap.parse_args())
image = cv2.imread(args["image"])
blur = cv2.blur(image, (5,5))

plt.subplot(131), plt.imshow(image), plt.title("Original")
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])

og = blur.reshape((blur.shape[0] * blur.shape[1], 3))

clusters = 3
#for clusters in range(2, 4):
clt = KMeans(n_clusters = clusters)
clt.fit(og)

hist = utils.centroid_histogram(clt)
bar = utils.plot_colors(hist, clt.cluster_centers_)

plt.subplot(133), plt.imshow(bar), plt.title("Bar Chart")
plt.imshow(bar)
plt.xticks([]), plt.yticks([])
plt.show()
