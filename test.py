from PIL import Image
from pix2tex.cli import LatexOCR

img = Image.open('math.png')
model = LatexOCR()
print(model(img))
