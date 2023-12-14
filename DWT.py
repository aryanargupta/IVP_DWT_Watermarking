import numpy as np
import pywt
import os
from PIL import Image
from scipy.fftpack import dct
from scipy.fftpack import idct

# Get the current directory of the script or module
current_directory = str(os.path.dirname(__file__))

# Input file names
input_image_name = 'imagetest1.jpg'
input_watermark_name = 'qrcodetest1.png'

def convert_image(input_image_name, size):
    # Open and resize the image
    img = Image.open('./pictures/' + input_image_name).resize((size, size), 1)
    
    # Convert the image to grayscale
    img = img.convert('L')
    
    # Save the converted image
    img.save('./dataset/' + input_image_name)

    # Convert the image to a NumPy array
    image_array = np.array(img.getdata(), dtype=float).reshape((size, size))
    
    # Display some elements of the image array
    print(image_array[0][0])
    print(image_array[10][10])

    return image_array

def process_coefficients(image_array, wavelet_model, decomposition_level):
    # Apply wavelet transformation to the image
    coefficients = pywt.wavedec2(data=image_array, wavelet=wavelet_model, level=decomposition_level)
    coefficients_list = list(coefficients)

    return coefficients_list

def embed_mod2(image_coefficients, watermark_coefficients, offset=0):
    # Embed the watermark into image coefficients with modulation of 2
    for i in range(len(watermark_coefficients)):
        for j in range(len(watermark_coefficients[i])):
            image_coefficients[i*2+offset][j*2+offset] = watermark_coefficients[i][j]

    return image_coefficients

def embed_mod4(image_coefficients, watermark_coefficients):
    # Embed the watermark into image coefficients with modulation of 4
    for i in range(len(watermark_coefficients)):
        for j in range(len(watermark_coefficients[i])):
            image_coefficients[i*4][j*4] = watermark_coefficients[i][j]

    return image_coefficients

def embed_watermark(watermark_array, original_image_dct):
    watermark_array_size = watermark_array[0].__len__()
    watermark_flat = watermark_array.ravel()
    index = 0

    for x in range(0, original_image_dct.__len__(), 8):
        for y in range(0, original_image_dct.__len__(), 8):
            if index < watermark_flat.__len__():
                subdct = original_image_dct[x:x+8, y:y+8]
                subdct[5][5] = watermark_flat[index]
                original_image_dct[x:x+8, y:y+8] = subdct
                index += 1

    return original_image_dct

def apply_dct(image_array):
    size = image_array[0].__len__()
    all_subdct = np.empty((size, size))
    
    # Apply DCT to non-overlapping 8x8 blocks
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            subpixels = image_array[i:i+8, j:j+8]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            all_subdct[i:i+8, j:j+8] = subdct

    return all_subdct

def inverse_dct(all_subdct):
    size = all_subdct[0].__len__()
    all_subidct = np.empty((size, size))
    
    # Apply inverse DCT to reconstruct the image
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            subidct = idct(idct(all_subdct[i:i+8, j:j+8].T, norm="ortho").T, norm="ortho")
            all_subidct[i:i+8, j:j+8] = subidct

    return all_subidct

def get_watermark(dct_watermarked_coefficients, watermark_size):
    subwatermarks = []

    for x in range(0, dct_watermarked_coefficients.__len__(), 8):
        for y in range(0, dct_watermarked_coefficients.__len__(), 8):
            coeff_slice = dct_watermarked_coefficients[x:x+8, y:y+8]
            subwatermarks.append(coeff_slice[5][5])

    watermark = np.array(subwatermarks).reshape(watermark_size, watermark_size)

    return watermark

def recover_watermark(image_array, wavelet_model='haar', decomposition_level=1):
    # Process coefficients of the watermarked image
    watermarked_image_coefficients = process_coefficients(image_array, wavelet_model, level=decomposition_level)
    
    # Apply DCT to the LL sub-band coefficients
    dct_watermarked_coefficients = apply_dct(watermarked_image_coefficients[0])
    
    # Get the watermark from the DCT coefficients
    watermark_array = get_watermark(dct_watermarked_coefficients, 128)
    
    # Convert the watermark array to uint8 type
    watermark_array = np.uint8(watermark_array)

    # Save the recovered watermark as an image
    img = Image.fromarray(watermark_array)
    img.save('./result/recovered_watermark.jpg')

def print_image_from_array(image_array, name):
    # Clip values between 0 and 255 and convert to uint8 for image saving
    image_array_copy = image_array.clip(0, 255)
    image_array_copy = image_array_copy.astype("uint8")
    
    # Save the image
    img = Image.fromarray(image_array_copy)
    img.save('./result/' + name)

def watermark_and_recover(wavelet_model='haar', decomposition_level=1):
    # Convert input images to NumPy arrays
    input_image_array = convert_image(input_image_name, 2048)
    watermark_array = convert_image(input_watermark_name, 128)

    # Process coefficients of the input image
    image_coefficients = process_coefficients(input_image_array, wavelet_model, level=decomposition_level)
    
    # Apply DCT to the LL sub-band coefficients
    dct_array = apply_dct(image_coefficients[0])
    
    # Embed the watermark into the DCT coefficients
    dct_array = embed_watermark(watermark_array, dct_array)
    image_coefficients[0] = inverse_dct(dct_array)

    # Reconstruct the watermarked image
    watermarked_image_array = pywt.waverec2(image_coefficients, wavelet_model)
    
    # Save the watermarked image
    print_image_from_array(watermarked_image_array, 'image_with_watermark.jpg')

    # Recover the watermark from the watermarked image
    recover_watermark(image_array=watermarked_image_array, wavelet_model=wavelet_model, decomposition_level=decomposition_level)

# Test the watermarking and recovery process
watermark_and_recover(wavelet_model="test_wavelet", decomposition_level=
