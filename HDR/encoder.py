import colour
import numpy as np

def calculate_prediction_error(E):
    predicted_E = np.zeros_like(E)
    predicted_E[1:, 1:] = np.minimum(E[:-1, 1:], E[1:, :-1])

    PE = E - predicted_E
    abs_PE = np.abs(PE)

    PE_prime = np.zeros_like(E)
    for i in range(1, PE.shape[0]):
        for j in range(1, PE.shape[1]):
            if PE[i, j-1] <= min(PE[i-1, j], PE[i, j+1]):
                PE_prime[i, j] = max(PE[i-1, j], PE[i, j+1])
            elif PE[i, j-1] >= max(PE[i-1, j], PE[i, j+1]):
                PE_prime[i, j] = min(PE[i-1, j], PE[i, j+1])
            else:
                PE_prime[i, j] = PE[i, j-1] + PE[i, j] - PE[i-1, j]
    return PE_prime

# Ask the user for the name of the HDR image they wish to process
image_name = input("Please enter the name of the HDR image to process (including file extension): ")
hdr_image_path = f'./HDR/HDR images/{image_name}'

# Read the image using the 'colour' library
hdr_image = colour.read_image(hdr_image_path)

# Check if the image has been read correctly
if hdr_image is None:
    print(f'Unable to read image: {hdr_image_path}')
    exit()

# Print the shape and type of the image for inspection
print(f'Image shape: {hdr_image.shape}')
print(f'Image type: {hdr_image.dtype}')

# Assuming the E channel is the fourth channel (if it exists)
if hdr_image.shape[2] == 4:
    E_channel = hdr_image[:, :, 3]

    # Apply the preprocessing function
    PE_prime = calculate_prediction_error(E_channel)

    # Now you can use PE_prime for further processing or save it as an image
else:
    print("The provided image is not in a valid RGBE format.")
