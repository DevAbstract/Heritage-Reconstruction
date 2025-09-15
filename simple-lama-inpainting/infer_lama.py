# infer_lama.py
from simple_lama_inpainting import SimpleLama
from PIL import Image

lama = SimpleLama()   # downloads model the first time

image = Image.open('/Users/gauravdevpandey/Projects/Heritage-Reconstruction/simple-lama-inpainting/dataset/dome.png').convert('RGB')
mask  = Image.open('/Users/gauravdevpandey/Projects/Heritage-Reconstruction/simple-lama-inpainting/dataset/mask.png').convert('L')  # single channel

# result = lama.inpaint(image, mask)  # returns PIL.Image
result = lama(image, mask)
result.save('image_inpainted.png')
print("Saved image_inpainted.png")

