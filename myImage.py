from PIL import Image
import numpy as np
import math as math

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
    def unsharp(self):
        unsharpKernel = np.array([[-1/6 , -2/3 ,-1/6],
                                  [-2/3,13/3,-2/3],
                                  [-1/6, -2/3,-1/6]]).reshape(3,3,1)
        self.array = self.svertka(unsharpKernel)
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
    def gabor(self, sigma, gamma, theta, Lambda, psi):
        sigma_x = sigma
        sigma_y = float(sigma) / gamma

        # Bounding box
        nstds = 3  # Number of standard deviation sigma
        xmax = max(abs(nstds * sigma_x * math.cos(theta)), abs(nstds * sigma_y * math.sin(theta)))
        xmax = np.ceil(max(1, xmax))
        ymax = max(abs(nstds * sigma_x * math.sin(theta)), abs(nstds * sigma_y * math.cos(theta)))
        ymax = np.ceil(max(1, ymax))
        xmin = -xmax
        ymin = -ymax
        y, x = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

        # Rotation
        x_theta = x * math.cos(theta) + y * math.sin(theta)
        y_theta = -x * math.sin(theta) + y * math.cos(theta)

        gb = np.exp(-0.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(
            2 * math.pi * x_theta / Lambda + psi)
        #print(gb.reshape())
        #print(xmax,ymax)
        arrayTmp = self.svertka(gb.reshape((int)(xmax)*2 +1,(int)(ymax)*2+1,1))
        self.array = arrayTmp
    def svertka(self, kernel):
        arrayTmp = np.copy(self.array)
        radius = kernel.shape[0] // 2
        for i in range(0, self.raws):
            for j in range(0, self.cals):
                window = self.get_square(i,j,radius)
                tmp = (window * kernel).sum(axis=(0, 1))
                tmp[tmp > 255] = 255
                tmp[tmp < 0] = 0
                arrayTmp[i, j] = tmp
        return arrayTmp
    def mse(self,image):
        self.gauss(1,1)
        tmp = (self.array - image.array)**2
        mse = np.sum(tmp) / (self.raws * self.cals)
        return mse
    def psnr(self,image):
        psnr = 10 *math.log10(255*255/self.mse(image))
        return psnr
    def ssim(self,image):
        self.gauss(3, 2)
        mean1 = self.array.mean()
        mean2 = image.array.mean()

        d1 = (self.array - mean1)**2
        d1 = np.sum(d1)/(self.raws * self.cals -1)

        d2 = (image.array - mean2) ** 2
        d2 = np.sum(d2) / (self.raws * self.cals - 1)

        cov = (self.array - mean1 )*(image.array - mean2)
        cov = np.sum(cov) / (self.raws * self.cals - 1)

        ssim = (2*mean1*mean2+1)*(2*cov**(1/2) + 1)/((mean1**2 + mean2**2 + 1)*(d1**2+d2**2 +1))
        return ssim
    def canny(self, sigma, high, low):
        self.gauss(sigma,1)
        self.sobelY()

    def bileteralKernelR(self,window,sigmaD,sigmaR):
        radius = 3*sigmaD
        kernelR = np.empty([radius * 2 + 1, radius * 2 + 1, 3]).reshape(radius*2+1,radius*2+1,3);
        for i in range(0, radius * 2 + 1):
            for j in range(0, radius * 2 + 1):
                kernelR[i, j] = np.exp(
                    -(window[i,j]- window[radius,radius])**2   / (2 * sigmaR * sigmaR)) / (2 * math.pi * sigmaR * sigmaR)
        return kernelR
    def mean (self,window):
        return window.sum(axis = (0,1)) / window.size
    def non_local_kernel(self,window,sigmaR):
        radius = 3*sigmaR
        kernelR = np.zeros([self.raws, self.cals, 3]).reshape(self.raws,self.cals,3);
        windowMean = self.mean(window)
        h = 2 *sigmaR * sigmaR
        hpi = 2 * math.pi * sigmaR * sigmaR
        for i in range(0, self.raws):
            for j in range(0, self.cals):
                vector = self.get_square(i,j,radius)
                module = windowMean - self.mean(vector)
                kernelR[i, j] = np.exp(
                    -(module)**2   / h) / hpi
        kernelR /= kernelR.sum(axis=(0, 1))
        return kernelR
    def non_local(self, sigmaR):
        radius = 3 * sigmaR
        arrayTmp = np.copy(self.array)
        q = 0
        t = 0
        for i in range(0, self.raws):
            for j in range(0, self.cals):
                q += 1
                if q/(self.raws*self.cals) > t:
                    print(t)
                    t+=0.01
                window = self.get_square(i,j,radius)
                kernel = self.non_local_kernel(window,sigmaR)
                arrayTmp[i, j] = (self.array * kernel).sum(axis=(0, 1))
                arrayTmp[ arrayTmp > 255] = 255
                arrayTmp[ arrayTmp < 0] = 0
        self.array = arrayTmp
    def get_square(self,i,j,radius):
        top = 1
        bot = 1
        left = 1
        right = 1
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
        return np.copy(window)
    def get_half_square(self,i,j,radius):
        top = 1
        bot = 1
        left = 1
        right = 1
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
        return np.copy(window)
    def bileteral(self, sigmaR, sigmaD):
        radius = 3 * sigmaD
        kernelGauss = np.empty([radius * 2 + 1, radius * 2 + 1, 1]);
        for i in range(0, radius * 2 + 1):
            for j in range(0, radius * 2 + 1):
                kernelGauss[i, j] = np.exp(
                    -((i - radius) * (i - radius) + (j - radius) * (j - radius)) / (2 * sigmaD * sigmaD)) / (2 * math.pi * sigmaD * sigmaD)

        arrayTmp = np.copy(self.array)
        for i in range(0, self.raws):
            for j in range(0, self.cals):
                window = self.get_square(i,j,radius)
                kernel = self.bileteralKernelR(window, sigmaD, sigmaR) * kernelGauss
                arrayTmp[i, j] = (window * kernel).sum(axis=(0, 1)) / kernel.sum(axis=(0, 1))
                arrayTmp[ arrayTmp > 255] = 255
                arrayTmp[ arrayTmp < 0] = 0
        self.array = arrayTmp