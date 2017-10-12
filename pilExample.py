import HYP_Utils
import HYP_Texture
import HYP_Material
import sys

from PIL import Image

scriptDir = HYP_Utils.GetDemoDir()

PIL_Version = Image.VERSION

img_filename = "%s/flower.jpg" % scriptDir
im = Image.open(img_filename)

imageW = im.size[0]
imageH = im.size[1]

TEXTURE_2D = 2
RGB_BYTE = 2
texId = HYP_Texture.Create(TEXTURE_2D, RGB_BYTE, imageW, imageH, 0)

matId = HYP_Material.GetId("plane1_mat")
HYP_Material.AddTexture(matId, texId)

if (im.mode == "RGB"):
  for y in range(0, imageH):
    for x in range(0, imageW):
      offset = y*imageW + x
      xy = (x, y)
      rgb = im.getpixel(xy)
      HYP_Texture.SetValueTex2DByteRgb(texId, offset, rgb[0], rgb[1], rgb[2])
elif (imout.mode == "L"):
  for y in range(0, imageH):
    for x in range(0, imageW):
      offset = y*imageW + x
      xy = (x, y)
      rgb = im.getpixel(xy)
      HYP_Texture.SetValueTex2DByteRgb(texId, offset, rgb, rgb, rgb)