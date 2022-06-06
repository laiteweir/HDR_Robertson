import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2 as cv

if __name__ == '__main__':
    img = cv.imread("hdr.exr", cv.IMREAD_UNCHANGED)
    intensity = np.dot(img, [19./256., 183./256., 54./256.])
    color = img / (intensity[:, :, np.newaxis] + 1e-8)
    intensity = intensity.astype(np.float32)
    large = cv.bilateralFilter(intensity, 9, 75, 75)
    detail = intensity / large
    log_mean = lambda img: np.exp(np.mean(np.log(img)))
    Lm = large * 0.2 / log_mean(large)
    Ld = 255 * Lm / (1 + Lm)
    temp = Ld * detail
    img_out = color * temp[:, :, np.newaxis]
    cv.imwrite("bilateral.png", img_out)