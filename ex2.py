import argparse
import sys

from myImage import MyImage


def createParser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    mse_parser = subparsers.add_parser('mse')
    mse_parser.add_argument("inputImage")
    mse_parser.add_argument("inputImage2")

    gabor_parser = subparsers.add_parser('gabor')
    gabor_parser.add_argument("sigma", type=int)
    gabor_parser.add_argument("gamma", type=int)
    gabor_parser.add_argument("theta", type=int)
    gabor_parser.add_argument("lam", type=int)
    gabor_parser.add_argument("psi", type=int)
    gabor_parser.add_argument("inputImage")

    psnr_parser = subparsers.add_parser('psnr')
    psnr_parser.add_argument("inputImage")
    psnr_parser.add_argument("inputImage2")

    psnr_parser = subparsers.add_parser('ssim')
    psnr_parser.add_argument("inputImage")
    psnr_parser.add_argument("inputImage2")

    canny_parser = subparsers.add_parser('canny')
    canny_parser.add_argument("sigma",type = int)
    canny_parser.add_argument("thr_high",type = float)
    canny_parser.add_argument("thr_low", type = float)
    canny_parser.add_argument("inputImage")

    #canny_parser.add_argument("outputImage")


    return parser
parser = createParser()
namespace = parser.parse_args(sys.argv[1:])
print(namespace)

if namespace.command == "mse":
    image = MyImage(namespace.inputImage)
    image2 = MyImage(namespace.inputImage2)
    print(image.mse(image2))
elif namespace.command == "psnr":
    image = MyImage(namespace.inputImage)
    image2 = MyImage(namespace.inputImage2)
    print(image.psnr(image2))
elif namespace.command == "ssim":
    image = MyImage(namespace.inputImage)
    image2 = MyImage(namespace.inputImage2)
    print(image.ssim(image2))
elif namespace.command == "canny":
    image = MyImage(namespace.inputImage)
    image.canny(namespace.sigma, namespace.thr_high, namespace.thr_low)
    image.view()
elif namespace.command == "gabor":
    image = MyImage(namespace.inputImage)
    image.gabor(namespace.sigma, namespace.gamma, namespace.theta, namespace.lam, namespace.psi )
    image.view()
