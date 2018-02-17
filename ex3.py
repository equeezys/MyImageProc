import argparse
import sys

from myImage import MyImage


def createParser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')


    bileteral_parser = subparsers.add_parser('bileteral')
    bileteral_parser.add_argument("sigma_d", type=int)
    bileteral_parser.add_argument("sigma_r", type=int)
    bileteral_parser.add_argument("inputImage")

    bileteral_parser = subparsers.add_parser('non_local')
    bileteral_parser.add_argument("sigma_r", type=int)
    bileteral_parser.add_argument("inputImage")
    bileteral_parser.add_argument("outputImage")

    gabor_parser = subparsers.add_parser('gabor')
    gabor_parser.add_argument("sigma", type=int)
    gabor_parser.add_argument("gamma", type=int)
    gabor_parser.add_argument("theta", type=int)
    gabor_parser.add_argument("lam", type=int)
    gabor_parser.add_argument("psi", type=int)
    gabor_parser.add_argument("inputImage")

    # canny_parser.add_argument("outputImage")


    return parser


parser = createParser()
namespace = parser.parse_args(sys.argv[1:])
print(namespace)


if namespace.command == "bileteral":
    image = MyImage(namespace.inputImage)
    image.bileteral(namespace.sigma_r, namespace.sigma_d)
    image.view()
elif namespace.command == "gabor":
    image = MyImage(namespace.inputImage)
    image.gabor(namespace.sigma, namespace.gamma, namespace.theta, namespace.lam, namespace.psi )
    image.view()
elif namespace.command == "non_local":
    image = MyImage(namespace.inputImage)
    image.non_local(namespace.sigma_r)
    image.view()
    image.save(namespace.outputImage)
