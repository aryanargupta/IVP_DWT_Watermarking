import numpy as np
from PIL import Image

def subtract_images(image1_path, image2_path, save_path):
    # Open images
    img1 = Image.open(image1_path).convert('L')  # Convert to grayscale
    img2 = Image.open(image2_path).convert('L')

    # Convert images to NumPy arrays
    array1 = np.array(img1)
    array2 = np.array(img2)

    # Ensure both arrays have the same shape
    if array1.shape != array2.shape:
        raise ValueError("Images must have the same dimensions.")

    # Subtract matrices
    result_array = array1 - array2

    # Optionally, save the result as a new image
    result_image = Image.fromarray(result_array)
    result_image.save(save_path)

# Example usage
image1_path = './dataset/imagetest1.jpg'
image2_path = './result/image_with_watermark.jpg'
save_path = './result/matrix_subtraction_result.jpg'

subtract_images(image1_path, image2_path, save_path)
