from PIL import Image
import numpy as np
import math as math
import sys
import argparse


def createParser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    mirror_parser = subparsers.add_parser('mirror')
    mirror_parser.add_argument("axe", choices=["x", "y"])

    sobel_parser = subparsers.add_parser('sobel')
    sobel_parser.add_argument("axe", choices=["x", "y"])

    median_parser = subparsers.add_parser('median')
    median_parser.add_argument("radius", type=int)

    gradient_parser = subparsers.add_parser('gradient')
    gradient_parser.add_argument("sigma", type=int)

    rotate_parser = subparsers.add_parser('rotate')
    rotate_parser.add_argument("direction", choices=["cw", "ccw"])
    rotate_parser.add_argument("angle", type=int)

    gauss_parser = subparsers.add_parser('gauss')
    gauss_parser.add_argument("sigma", type=int)
    gauss_parser.add_argument("gamma", type=float, default=1.0)

    parser.add_argument("inputImage")
    parser.add_argument("outputImage")

    return parser


class MyImage:
    def __init__(self, srcimage):
        self.image = Image.open(srcimage)
        self.array = np.asarray(self.image)
        self.raws = self.array.shape[0]
        self.cals = self.array.shape[1]

    def mirrorX(self):
        self.array = self.array[:, ::-1]

    def mirrorY(self):
        self.array = self.array[::-1]

    def rotate(self,a):
        a = 180 * a / math.pi
        kernelRotate = np.array([[math.cos(a), -math.sin(a)],
                                 [math.sin(a), math.cos(a)]]).reshape(2, 2, 1)


    def rotateCW(self, a):
        for a in range(a // 90):
            self.array = self.array.transpose((1, 0, 2))
            self.array = self.array[:, ::-1]

    def rotateCCW(self, a):
        for a in range(a // 90):
            self.array = self.array.transpose((1, 0, 2))
            self.array = self.array[::-1]

    def view(self):
        newimage = Image.fromarray(self.array)
        newimage.show()

    def save(self, path):
        newimage = Image.fromarray(self.array)
        newimage.save(path)

    def mediana(self, radius):
        top = 1
        bot = 1
        left = 1
        right = 1
        arrayTmp = np.copy(self.array)
        for i in range(0, self.raws):
            for j in range(0, self.cals):
                if i in range(0, radius):
                    top = 0
                else:
                    if i in range(self.raws - radius, self.raws):  # проверка граничных условий
                        bot = 0
                if j in range(0, radius):
                    left = 0
                else:
                    if j in range(self.cals - radius, self.cals):
                        right = 0
                arrayTmp[i, j] = np.median(
                    self.array[i - radius * top:i + radius * bot + 1:, j - radius * left:j + radius * right + 1:],
                    (0, 1))
                if (top * bot == 0):
                    top = 1
                    bot = 1
                if (left * right == 0):
                    left = 1
                    right = 1
        self.array = arrayTmp

    def sobelX(self):
        kernelSobelX = np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]]).reshape(3, 3, 1)
        self.array = self.svertka(kernelSobelX)
        self.array[self.array < 0] += 128

    def sobelY(self):
        kernelSobelY = np.array([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]]).reshape(3, 3, 1)
        self.array = self.svertka(kernelSobelY)
        self.array[self.array < 0] += 128  # серая подложка

    def gamma(self, g):
        arrayTmp = np.copy(self.array)
        for i in range(0, self.raws):
            for j in range(0, self.cals):
                arrayTmp[i, j] = ((self.array[i, j] / 255) ** g) * 255
        self.array = arrayTmp

    def gauss(self, s, g):
        radius = 3 * s
        kernelGauss = np.empty([radius * 2 + 1, radius * 2 + 1, 1]);
        self.gamma(1/g)
        for i in range(0, radius * 2 + 1):
            for j in range(0, radius * 2 + 1):
                kernelGauss[i, j] = math.exp(
                    -((i - radius) * (i - radius) + (j - radius) * (j - radius)) / (2 * s * s)) / (2 * math.pi * s * s)
        self.array = self.svertka(kernelGauss)
        self.gamma(g)

    def gradient(self, s):
        radius = 3 * s
        kernelGaussX = np.empty([radius * 2 + 1, radius * 2 + 1, 1])
        kernelGaussY = np.empty([radius * 2 + 1, radius * 2 + 1, 1])
        for i in range(0, radius * 2 + 1):
            for j in range(0, radius * 2 + 1):
                kernelGaussX[i, j] = (-1.0 * (i - radius) / s * s) * math.exp(
                    -((i - radius) * (i - radius) + (j - radius) * (j - radius)) / (2 * s * s)) / (2 * math.pi * s * s)
        kernelGaussY = kernelGaussX.transpose((1, 0, 2))
        arrayTmp = np.copy(self.array)
        gaussY = self.svertka(kernelGaussY)
        gaussX = self.svertka(kernelGaussX)
        for i in range(0, self.raws):
            for j in range(0, self.cals):
                for q in range(0, 3):
                    arrayTmp[i, j, q] = int(
                        math.sqrt(gaussX[i, j, q] * gaussX[i, j, q] + gaussY[i, j, q] * gaussY[i, j, q]))
        arrayTmp[arrayTmp > 255] = 255
        arrayTmp[arrayTmp < 0] = 0
        self.array = arrayTmp

    def svertka(self, kernel):
        top = 1
        bot = 1
        left = 1
        right = 1
        arrayTmp = np.copy(self.array)
        radius = kernel.shape[0] // 2
        for i in range(0, self.raws):
            for j in range(0, self.cals):
                if i in range(0, radius):
                    top = 0
                else:
                    if i in range(self.raws - radius, self.raws):  # проверка граничных условий
                        bot = 0
                if j in range(0, radius):
                    left = 0
                else:
                    if j in range(self.cals - radius, self.cals):
                        right = 0

                window = self.array[i - radius * top:i + radius * bot + 1:, j - radius * left:j + radius * right + 1:]

                if (bot == 0):
                    window = np.vstack((window, window[:0:-1]))
                    bot = 1
                if (top == 0):
                    window = np.vstack((window[:0:-1], window))
                    top = 1
                if (right == 0):
                    window = np.hstack((window, window[:, :0:-1]))
                    right = 1
                if (left == 0):
                    window = np.hstack((window[:, :0:-1], window))
                    left = 1
                arrayTmp[i, j] = (window * kernel).sum(axis=(0, 1))
        arrayTmp[arrayTmp > 255] = 255
        arrayTmp[arrayTmp < 0] = 0
        return arrayTmp


parser = createParser()
namespace = parser.parse_args(sys.argv[1:])
print(namespace)

image = MyImage(namespace.inputImage)
if namespace.command == "mirror":
    if namespace.axe == "x":
        image.mirrorX()
    else:
        image.mirrorY()
elif namespace.command == "sobel":
    if namespace.axe == "x":
        image.sobelX()
    else:
        image.sobelY()
elif namespace.command == "median":
    image.mediana(namespace.radius)
elif namespace.command == "gradient":
    image.gradient(namespace.sigma)
elif namespace.command == "rotate":
    if namespace.direction == "cw":
        image.rotate(namespace.angle) #rotateCW
    else:
        image.rotateCCW(namespace.angle)
elif namespace.command == "gauss":
    image.gauss(namespace.sigma, namespace.gamma)
image.view()
image.save(namespace.outputImage)
