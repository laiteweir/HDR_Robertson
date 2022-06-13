import argparse
import numpy as np
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument("-P", "--numpictures", type=int, help="(int) Number of source images", default=8)
args = parser.parse_args()

if __name__ == '__main__':
    P = args.numpictures
    images = [cv.imread(f"./images/{j}.png") for j in range(1, P+1)]
    images_out = []
    for j in range(P):
        images_out.append(cv.resize(images[j], (900, 600), interpolation=cv.INTER_AREA))
    for j in range(1, P+1):
        cv.imwrite(f'./temp/{j}.png', images_out[j-1])