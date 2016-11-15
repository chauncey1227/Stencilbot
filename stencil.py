from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import matplotlib.cm as cm
import numpy as np
import argparse
import cv2
import utils


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image to stencil")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

r = 100.0 / image.shape[1]
dim = (100, int(image.shape[0] * r))

image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

plt.figure("Original")
plt.axis("off")
plt.imshow(image)

# reshape the image to be a list of pixels
image_array = image.reshape((image.shape[0] * image.shape[1], 3))

for clusters in range(4, 5):
    clt = KMeans(n_clusters = clusters)
    clt.fit(image_array)

    pixel_and_cluster = zip(image_array, clt.labels_)
    for i, (pixel, cluster_num) in enumerate( pixel_and_cluster):
        image_array[i] = clt.cluster_centers_[cluster_num]

    plt.figure("New")
    plt.axis("off")
    plt.imshow( np.array(image_array).reshape(100,100,3) )

    plt.show()
