import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from plot_tools import plot_individual_tensor, plot_full_result, add_grid_lines
from model_utils import try_sam, determine_most_likely, evaluation_metrics, append_results_to_csv
import os
import csv
import re

def train_tile_model(sam, image, prompt, bt, tt, object_being_predicted, save_images = False, save_results = False):
    """
    Train tile model and evaluate it against a ground truth, saving the results and images.
    
    Parameters:
        sam (model): SAM model.
        image (str): Identifier for the image (without file extension).
        prompt (str): Prompt for segmentation.
        bt (float): Box threshold.
        tt (float): Text threshold.
        object_being_predicted (str): Feature being segmented ('Urban', 'Tree', 'Cloud', or 'Water').
        save_images (bool): Saves results images if True.
        save_results (bool): Saves model results if True.
    """
    # Paths for raw and ground truth images
    raw = f'../data/gamma/{image}.png'
    ground_truth = f'../data/labeled/{image}_colored.png'

    # Check valid response
    urban_prompt = ""; cloud_prompt = ""; tree_prompt = ""; water_prompt = "";
    if object_being_predicted == "Urban":
        urban_prompt = prompt
    elif object_being_predicted == "Water":
        water_prompt = promp
    elif object_being_predicted == "Tree":
        tree_prompt = prompt
    elif object_being_predicted == "Cloud":
        cloud_prompt = prompt
    else:
        print("object_being_predicted must be 'Urban', 'Water', 'Tree', or 'Cloud'.")
        return
   
    # Generate SAM masks
    tile_info = segment_tiles(sam, image, prompt, bt, tt, object_being_predicted)
    result_tensor, prediction = stitch_tiles(tile_info)
    prediction = add_grid_lines(prediction)
    height, width = result_tensor.shape
    
    # Extract performance metrics
    ground_truth_img = Image.open(ground_truth)
    ground_truth_img = ground_truth_img.crop((0, 0, width, height))
    cm, accuracy, weighted_f1, dice_scores, counts = evaluation_metrics(ground_truth_img, result_tensor)
    
    # Save results
    if save_results:
        file_path = '../results/tile_results.csv'
        append_results_to_csv(file_path, "tile", image, tt, bt, urban_prompt, cloud_prompt, tree_prompt, water_prompt, cm, accuracy, weighted_f1, dice_scores, counts)

    # Save plots
    if save_images:
        folder_path = f'../results/tile/{image}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    filename = f'../results/tile/{image}/tile_{prompt}_{object_being_predicted}_bt{bt}_tt{tt}.png'
    return plot_full_result(raw, ground_truth, prediction, filename, save_images)

def segment_tiles(model, image, text_prompt, box_threshold, text_threshold, object_being_predicted):
    """
    Apply SAM to individual tiles.

    Args:
    model (model): SAM model.
    image (str): Base name of the image files to be processed.
    text_prompt (str): Prompts for segmentation.
    box_threshold (float): Box threshold.
    text_threshold (float): Text threshold.
    object_being_predicted (str): Feature being segmented ('Urban', 'Tree', 'Cloud', or 'Water').

    Returns:
    list of tuples: Each tuple contains the indices (i_index, j_index) of tile and its associated mask.
    """
    tiles = []
    # Walk through directory to find relevant image tiles
    for root, dirs, files in os.walk('../data/tiles'):
        for file in files:
            if file.endswith(".png") and file.startswith(image):
                # Extract indices from filename using regular expression
                pattern = r"image_\d+_\d{4}-\d{2}-\d{2}_(\d)_(\d)\.png"
                match = re.search(pattern, file)
                i_index = int(match.group(1))
                j_index = int(match.group(2))
                image_path = os.path.join(root, file)
                raw_img = Image.open(image_path)
                width, height = raw_img.size
                
                # Apply SAM to get mask for tile
                mask = try_sam(model, image_path, text_prompt, box_threshold, text_threshold, height, width)
                mask = determine_most_likely([mask], [object_being_predicted])
                tiles.append((i_index, j_index, mask))
    return tiles

def stitch_tiles(tile_info, num_tiles = 5):
    """
    Combines multiple tensor tiles into a single large tensor.

    Args:
    tile_info (list of tuples): Each tuple contains the indices (i_index, j_index) of tile and its associated mask.
    num_tiles (int, optional): Number of tiles in one row/column of final stitched image. Assumes a square grid of tiles. Default is 5.

    Returns:
    tuple: A tuple containing two elements:
           - result_tensor (numpy.ndarray): Stitched tensor representing combined data from individual tiles.
           - result_image (numpy.ndarray): Stitched image created from combining individual tile images.
    """    
    result_tensor = None
    result_image = None
    
    for i_index, j_index, (tensor, image) in tile_info:

        # Extract dimensions and initialize tensors
        if result_tensor is None:
            tile_height, tile_width = tensor.shape
            tile_height_img, tile_width_img, channels = image.shape
            full_height= tile_height * num_tiles
            full_width = tile_width * num_tiles
            full_height_img = tile_height_img * num_tiles
            full_width_img = tile_width_img * num_tiles
            result_tensor = np.zeros((full_height, full_width), dtype = tensor.dtype)
            result_image = np.zeros((full_height_img, full_width_img, channels), dtype = image.dtype)

        # Place tile in image
        start_y = i_index * tile_height
        start_x = j_index * tile_width
        start_y_img = i_index * tile_height_img
        start_x_img = j_index * tile_width_img
        result_tensor[start_y:start_y + tile_height, start_x:start_x + tile_width] = tensor
        result_image[start_y_img:start_y_img + tile_height_img, start_x_img:start_x_img + tile_width_img] = image

    return result_tensor, result_image