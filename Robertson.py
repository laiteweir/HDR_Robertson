import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import argparse
import math
import numpy as np
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument("-P", "--numpictures", type=int, help="(int) Number of source images", default=18)
#parser.add_argument("-N", "--numpoints", type=int, help="(int) Number of sample points", default=25)
parser.add_argument("-w", "--wfunction", type=int, help="(int) Which weighting function to use, 1: triangle, 2: gaussian", default=1)
args = parser.parse_args()

def optimize_e(E, images, w, g, dt):
    P, M, N, D = images.shape
    for m in range(M):
        for n in range(N):
            for d in range(D):
                numerator = 0.
                denominator = 0.
                for j in range(P):
                    numerator += w[images[j, m, n, d]] * g[d, images[j, m, n, d]] * dt[j]
                    denominator += w[images[j, m, n, d]] * math.pow(dt[j], 2)
                E[m, n, d] = numerator / (denominator + 1e-8)

def optimize_g(E, images, g, dt):
    for d in range(3):
        for i in range(256):
            irradiance = 0.
            Em = np.asarray(images[:, :, :, d] == i).nonzero()
            num = Em[0].shape[0]
            if num == 0:
                continue
            for k in range(num):
                j = Em[0][k]
                m = Em[1][k]
                n = Em[2][k]
                #d = Em[3][k]
                irradiance += E[m, n, d] * dt[j]
            g[d, i] = irradiance / num
        g[d] /= g[d, 128]

def oneiter(E, images, w, g, dt):
    optimize_e(E, images, w, g, dt)
    optimize_g(E, images, g, dt)

if __name__ == '__main__':
    #initialize
    P = args.numpictures
    shutter_speed = np.array([0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    exposure_time = np.reciprocal(shutter_speed)
    w1 = np.zeros(256)
    for z in range(256):
        if z < 128:
            w1[z] = z
        else:
            w1[z] = 255 - z
    w1 = w1 / 127
    w2 = np.array([math.exp(-4 * math.pow(x - 127.5, 2) / math.pow(127.5, 2)) for x in range(-128, 128)])
    if args.wfunction == 1:
        w = w1
    else:
        w = w2
    x = np.arange(256, dtype=float)
    x = x * 1000000 / 255
    g = np.stack([x, x, x], axis=0)
    #I/O
    images = np.stack([cv.imread(f"./images2/{j}.png") for j in range(1, P+1)], axis=0)
    _, M, N, D = images.shape
    img = np.zeros((M, N, D))
    oneiter(img, images, w, g, exposure_time)
    oneiter(img, images, w, g, exposure_time)
    oneiter(img, images, w, g, exposure_time)
    oneiter(img, images, w, g, exposure_time)
    oneiter(img, images, w, g, exposure_time)
    optimize_e(img, images, w, g, exposure_time)
    img = img.astype(np.float32)
    cv.imwrite("hdr.exr", img)