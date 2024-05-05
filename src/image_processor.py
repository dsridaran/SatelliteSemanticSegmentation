import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.transform import rescale, resize
import skimage.exposure
from PIL import Image

def process_images(image, tiles = 5):
    """
    Process an image by applying a series of transformations and dividing it into smaller tiles.

    Parameters:
        image (str): The base name of the image file without file extension.
        tiles (int, optional): The number of tiles per row and column. Defaults to 5.

    This function performs the following steps:
    - Extracts the RGB channels from a TIFF image.
    - Applies gamma correction to the extracted image.
    - Splits the gamma-corrected image into smaller tiles and saves each tile.
    """
    # Extract RGB bands
    image_3 = extract_rgb(f'../data/raw/{image}.tiff', f'../data/rgb/{image}.png')

    # Perform gamma adjustment
    apply_gamma_correction(f'../data/rgb/{image}.png', f'../data/gamma/{image}.png')

    # Extract dimensions for each tile
    tile_size_x = image_3.shape[1] // tiles
    tile_size_y = image_3.shape[0] // tiles

    # Process each tile
    for i in range(tiles):
        for j in range(tiles):
            start_x = j * tile_size_x
            start_y = i * tile_size_y
            end_x = start_x + tile_size_x
            end_y = start_y + tile_size_y

            # Extract tile and save
            tile = image_3[start_y:end_y, start_x:end_x]
            plt.imsave(f'../data/tiles/{image}_{i}_{j}.png', tile)

def read_tiff(img_path, image_size = 1, resize_ratio = None, resizing = False, normalize = True):
    """
    Read a TIFF image, optionally resize, and normalize it.

    Parameters:
        img_path (str): Path to the TIFF image file.
        image_size (int, optional): The size to resize the image to. Defaults to 1 (no resizing).
        resize_ratio (float, optional): The ratio to rescale the image by. If provided, overrides image_size.
        resizing (bool, optional): Whether to resize the image. Defaults to False.
        normalize (bool, optional): Whether to normalize the image channels to range [0, 1]. Defaults to True.

    Returns:
        np.array: The processed image as a numpy array.
    """
    # Load image
    img = io.imread(img_path)
    img_F = img.copy()
    
    # Optionally resize image
    if resize_ratio:
        img_F = rescale(img, resize_ratio, anti_aliasing = True)
    if resizing:
        img_F = resize(img_F, (image_size, image_size), anti_aliasing = True)

    # Optionally normalize image
    if normalize:
        CHANNELS = range(12)
        img_F = np.dstack([skimage.exposure.rescale_intensity(img_F[:, :, c], out_range = (0, 1)) for c in CHANNELS])
    return img_F

def extract_rgb(image_path, output_path):
    """
    Extract RGB channels from a TIFF image and save as a PNG image.

    Parameters:
        image_path (str): Path to the TIFF image file.
        output_path (str): Path where the RGB image will be saved as a PNG.

    Returns:
        np.array: The extracted RGB image as a numpy array.
    """
    # Extract RGB channels and save
    img = read_tiff(image_path, normalize = True)
    image_3 = img[:, :, 1:4]
    plt.imsave(output_path, image_3)
    return image_3

def apply_gamma_correction(image_path, output_path = None, gamma = 0.5):
    """
    Apply gamma correction to an image.

    Parameters:
        image_path (str): Path to the image file.
        output_path (str): Path where the gamma-corrected image will be saved.
        gamma (float, optional): Gamma correction factor. Defaults to 0.5.
    """
    # Open and normalize image
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    img_normalized = img_np / 255.0
    
    # Apply gamma correction and save
    gamma_corrected = np.power(img_normalized, gamma)
    img_gamma = np.uint8(gamma_corrected * 255)
    img_gamma_pil = Image.fromarray(img_gamma)
    img_gamma_pil.save(output_path)