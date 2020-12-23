import numpy
import math as m
from scipy import ndimage
from scipy.ndimage.filters import convolve


def operator_sobel(img):
    grad_y = numpy.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float)
    grad_x = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    gorizontal = convolve(img, grad_x)
    vertical = convolve(img, grad_y)
    g_res = numpy.hypot(gorizontal, vertical)
    g_res = g_res / g_res.max() * 255
    theta = numpy.arctan2(vertical, gorizontal)
    return g_res, theta


def leave_pacification(image, teta):
    a, b = image.shape
    value = numpy.zeros(image.shape, int, 'C')
    corner = teta * 180. / 3.1415
    corner[corner < 0] += 180
    i = 1
    while i < a - 1:
        j = 1
        while j < b - 1:
            qt = 255
            rt = 255

            if (m.degrees(0) <= corner[i, j] < m.degrees(m.pi/8)) or (m.degrees(m.pi/8 * 7) <= corner[i, j] <= m.degrees(m.pi)):
                qt = image[i, j + 1]
                rt = image[i, j - 1]

            elif m.degrees(m.pi/8) <= corner[i, j] < m.degrees(m.pi/8 * 3):
                qt = image[i + 1, j - 1]
                rt = image[i - 1, j + 1]

            elif m.degrees(m.pi/8 * 3) <= corner[i, j] < m.degrees(m.pi/8 * 5):
                qt = image[i + 1, j]
                rt = image[i - 1, j]

            elif m.degrees(m.pi/8 * 5) <= corner[i, j] < m.degrees(m.pi/8 * 7):
                qt = image[i - 1, j - 1]
                rt = image[i + 1, j + 1]

            if (image[i, j] >= qt) and (image[i, j] >= rt):
                value[i, j] = image[i, j]
            else:
                value[i, j] = 0

            j += 1
        i += 1
    return value


def gauss(sz, sgm=1):
    sz = int(sz) // 2
    x, y = numpy.mgrid[-sz:sz + 1, -sz:sz + 1]
    ideal = 1 / (2 * 3.1415 * numpy.square(sgm))
    g = numpy.exp(-0.5 * (numpy.square(x) + numpy.square(y)) / numpy.square(sgm)) * ideal
    return g


def thresholds(image, low_pix, heavy_pix, threshold1, threshold2):
    threshold2 = image.max() * threshold2
    threshold1 = threshold2 * threshold1
    result = numpy.zeros(image.shape, int)
    heavy_i, heavy_j = numpy.where(image >= threshold2)
    low_i, low_j = numpy.where((image <= threshold2) & (image >= threshold1))
    result[heavy_i, heavy_j] = heavy_pix
    result[low_i, low_j] = low_pix
    return result


def dependence(image, low_pix, heavy_pix):
    a, b = image.shape
    low = low_pix
    heavy = heavy_pix
    i = 1
    while i < a - 1:
        j = 1
        while j < b - 1:
            if image[i, j] == low:
                if ((image[i + 1, j - 1] or image[i + 1, j] or image[i + 1, j + 1]
                        or image[i, j - 1] or image[i, j + 1]
                        or image[i - 1, j - 1] or image[i - 1, j]
                        or image[i - 1, j + 1]) == heavy):
                    image[i, j] = heavy
                else:
                    image[i, j] = 0
            j += 1
        i += 1
    return image


def shades_gray(rgb):
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray_col = 0.2989 * red + 0.5870 * green + 0.1140 * blue
    return gray_col


def algorithm_canny(image):
    image = shades_gray(image)
    image_flatten = convolve(image, gauss(5, 1))
    rug_grad, rug_teta = operator_sobel(image_flatten)
    pacification_img = leave_pacification(rug_grad, rug_teta)
    image_limit = thresholds(pacification_img, low_pix=75, heavy_pix=255, threshold1=0.05, threshold2=0.15)
    result_image = dependence(image_limit, low_pix=75, heavy_pix=255)
    return result_image
