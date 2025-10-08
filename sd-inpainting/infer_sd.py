from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch, os
# import tkinter as tk
# from tkinter import filedialog


# root = tk.Tk()
# root.withdraw()

# # Ask for image
# image_path = filedialog.askopenfilename(
#     title="Select Image",
#     filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
# )

# # Ask for mask
# mask_path = filedialog.askopenfilename(
#     title="Select Mask",
#     filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
# )

image_path = "./dataset/dome.png"
mask_path  = "./dataset/mask.png"
# output_path = "/outputs/sd_output8.png"

output_dir = os.path.join(os.path.dirname(__file__), "outputs")
base_name = "sd_output"
ext = ".png"

# find the next available filename
i = 1
while True:
    output_path = os.path.join(output_dir, f"{base_name}{i}{ext}")
    if not os.path.exists(output_path):
        break
    i += 1

os.makedirs(output_dir, exist_ok=True)

# Load pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")  # use ).to("mps") for mac devices
# Load inputs
image = Image.open(image_path).convert("RGB")
mask  = Image.open(mask_path).convert("RGB")

# Prompt-based reconstruction
prompt = "Reconstruct the missing dome with sandstone texture and Mughal-style architecture details."
# prompt = "Reconstruct the missing part of the ancient coin, restoring it realistically with consistent texture, metallic surface, engravings, and fine details. The design should seamlessly blend with the existing coin, maintaining symmetry, patina, and historical accuracy."
# result = pipe(prompt=prompt, image=image, mask_image=mask, guidance_scale=7.5, num_inference_steps=40)
result = pipe(prompt=prompt, image=image, mask_image=mask, guidance_scale=7.5, num_inference_steps=40)
result.images[0].save(output_path)
print(f"✅ Saved Stable Diffusion result to {output_path}")
