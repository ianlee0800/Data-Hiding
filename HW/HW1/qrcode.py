from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to prompt yes/no question
def ask_yes_no(question):
    response = input(f"{question} (y/n): ").strip().lower()
    while response not in ("y", "n"):
        response = input("Please enter 'y' for yes or 'n' for no: ").strip().lower()
    return response == 'y'

# Prompt the user for the image filename (without the extension)
image_filename = input("Please enter the name of the image file to watermark: ")

# Paths
qr_code_path = "./HW/HW1/images/qrcode.png"
image_path = f"./HW/HW1/images/{image_filename}.tiff"
output_path = "./HW/HW1/with_qrcode/"

# Load the QR code and the image
qr_code = Image.open(qr_code_path).convert("RGBA")
image = Image.open(image_path).convert("RGBA")

# Ensure QR code is not larger than the image
if qr_code.size[0] > image.size[0] or qr_code.size[1] > image.size[1]:
    print("The QR code is larger than the image. Resizing the QR code...")
    qr_code = qr_code.resize((image.size[0]//10, image.size[1]//10))

# Calculate position for the QR code on the bottom-right
position = (image.size[0] - qr_code.size[0], image.size[1] - qr_code.size[1])

# Function to blend images
def blend_images(image, qr_code, position, alpha):
    x, y = position
    # Crop the part of the image where the QR code will be placed
    image_part = image.crop((x, y, x+qr_code.size[0], y+qr_code.size[1]))
    # Blend with the QR code
    blended_part = Image.blend(image_part, qr_code, alpha=alpha)
    # Paste the blended part back to the image
    image.paste(blended_part, position)
    return image

# Ask the user for alpha value
alpha = float(input("Please enter the alpha value (between 0 and 1): "))

# Blend the image
watermarked_image = blend_images(image, qr_code, position, alpha)

# Ask if the user wants to display the image
if ask_yes_no("Would you like to display the watermarked image?"):
    plt.imshow(watermarked_image)
    plt.axis('off')  # Turn off the axis
    plt.show()

# Ask if the user wants to save the image
if ask_yes_no("Would you like to save the watermarked image?"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Format the filename to include the alpha value with two decimal places
    alpha_str = "{:.2f}".format(alpha).replace('.', '_')
    output_filename = f"{image_filename}_watermarked_alpha{alpha_str}.tiff"
    watermarked_image.save(os.path.join(output_path, output_filename))
    print(f"Watermarked image saved to {output_path}/{output_filename}")
