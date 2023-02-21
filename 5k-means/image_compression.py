import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import cv2

if __name__ == "__main__":
    image = plt.imread("data\\bird_small.png")  # (128,128,3)

    image = image.reshape((128 * 128, 3))
    kmeans = KMeans(n_clusters=16, random_state=0).fit(image)

    new_image = []
    for i in kmeans.labels_:
        new_image.append(list(kmeans.cluster_centers_[i, :]))  # 此处簇中心存储的是像素簇的中心，而不是位置簇中心
    new_image = np.array(new_image)
    new_image = new_image.reshape(128, 128, 3)
    plt.imshow(new_image)
    plt.axis('off')
    plt.title("new image")
    plt.show()

    plt.imsave("data\\2.png", new_image)

    # plt.subplot(1, 2, 2)
    # compressed_image, colors = compress(image, 16)
    # print(compressed_image.shape, colors.shape)
    # plt.imshow(compressed_format_to_normal_format(compressed_image, colors))
    # plt.axis('off')
    # plt.title("compressed image")
    # plt.show()
