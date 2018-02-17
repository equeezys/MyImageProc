from myImage import MyImage
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

    unsharp_parser = subparsers.add_parser('unsharp')

    parser.add_argument("inputImage")
    parser.add_argument("outputImage")

    return parser

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
elif namespace.command == "unsharp":
    image.unsharp()
image.view()
image.save(namespace.outputImage)
