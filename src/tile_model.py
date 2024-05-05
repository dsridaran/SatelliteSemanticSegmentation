import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from plot_tools import plot_individual_tensor, add_grid_lines
from model_utils import try_sam, determine_most_likely, evaluation_metrics
import os
import csv
import re

def train_tile_model(sam, image, text_prompt, box_threshold, text_threshold, object_being_predicted):

    # Create output folder
    folder_path = f'../results/tile/{image}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Check valid response
    if object_being_predicted not in ["Urban", "Water", "Tree", "Cloud"]:
        print("object_being_predicted must be 'urban', 'water', 'tree', or 'cloud'.")
        return

    # Create result file
    file_path = '../results/tile_results.csv'
    header = ['image', 'text_threshold', 'box_threshold', 'prompt', 'object_being_predicted', 'accuracy', 'weighted_f1', 'dice', 'y_true', 'y_pred']
    if not os.path.isfile(file_path):
        with open(file_path, 'w', newline = '') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    # Process individual tiles
    tile_info = segment_tiles(sam, image, text_prompt, box_threshold, text_threshold, object_being_predicted)

    # Stitch individual tiles and normalize output
    result_tensor, result_image = stitch_tiles(tile_info)
    height, width = result_tensor.shape

    # Plot raw image
    fig, axes = plt.subplots(1, 3, figsize = (18, 6), constrained_layout = True)
    fig.subplots_adjust(top = 0.9)
    img = mpimg.imread(f'../data/gamma/{image}.png')
    img = img[:height, :width]
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title('Raw Image')

    # Plot ground truth
    ground_truth_img = mpimg.imread(f'../data/labeled/{image}_colored.png')
    ground_truth_img = ground_truth_img[:height, :width]
    axes[1].imshow(ground_truth_img)
    axes[1].axis('off')
    axes[1].set_title('Manually Labeled Image')

    # Plot ensemble prediction
    processed_image = add_grid_lines(result_image.copy())
    axes[2].imshow(processed_image)
    axes[2].axis('off') 
    axes[2].set_title('Individual Prediction')
    fig.savefig(f'../results/tile/{image}/tile_{text_prompt}_{object_being_predicted}_bt{box_threshold}_tt{text_threshold}.png')
    plt.clf()
    plt.close(fig)    
    
    # Extract performance metrics
    ground_truth_img = Image.open(f'../data/labeled/{image}_colored.png')
    ground_truth_img = ground_truth_img.crop((0, 0, width, height))
    cm, accuracy, weighted_f1, dice_scores, y_t_u, y_t_c, y_t_t, y_t_w, y_p_u, y_p_c, y_p_t, y_p_w = evaluation_metrics(ground_truth_img, result_tensor)

    # Append results
    if object_being_predicted == "Urban":
        y_true = y_t_u; y_pred = y_p_u; dice = dice_scores[0]
    elif object_being_predicted == "Cloud":
        y_true = y_t_c; y_pred = y_p_c; dice = dice_scores[1]
    elif object_being_predicted == "Tree":
        y_true = y_t_t; y_pred = y_p_t; dice = dice_scores[2]
    elif object_being_predicted == "Water":
        y_true = y_t_w; y_pred = y_p_w; dice = dice_scores[3]

    data_to_append = [image, text_threshold, box_threshold, text_prompt, object_being_predicted, accuracy, weighted_f1, dice, y_true, y_pred]
    with open(file_path, 'a', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(data_to_append)

    return 

def segment_tiles(model, image, text_prompt, box_threshold, text_threshold, object_being_predicted):
    tiles = []
    for root, dirs, files in os.walk('../data/tiles'):
        for file in files:
            if file.endswith(".png") and file.startswith(image):
                pattern = r"image_\d+_\d{4}-\d{2}-\d{2}_(\d)_(\d)\.png"
                match = re.search(pattern, file)
                i_index = int(match.group(1))
                j_index = int(match.group(2))
                image_path = os.path.join(root, file)
                raw_img = Image.open(image_path)
                width, height = raw_img.size
                mask = try_sam(model, image_path, text_prompt, box_threshold, text_threshold, height, width)
                mask = determine_most_likely([mask], [object_being_predicted])
                tiles.append((i_index, j_index, mask))
    return tiles

def stitch_tiles(tile_info, num_tiles = 5):
    
    # Initialize full image
    result_tensor = None
    result_image = None

    # Iterate through tiles
    for i_index, j_index, (tensor, image) in tile_info:

        # Check if full_image is initialized
        if result_tensor is None:
            
            # Extract dimensions
            tile_height, tile_width = tensor.shape
            tile_height_img, tile_width_img, channels = image.shape
            full_height= tile_height * num_tiles
            full_width = tile_width * num_tiles
            full_height_img = tile_height_img * num_tiles
            full_width_img = tile_width_img * num_tiles
            
            # Initialize tensors
            result_tensor = np.zeros((full_height, full_width), dtype = tensor.dtype)
            result_image = np.zeros((full_height_img, full_width_img, channels), dtype = image.dtype)

        # Calculate starting indices for placing the tile
        start_y = i_index * tile_height
        start_x = j_index * tile_width
        start_y_img = i_index * tile_height_img
        start_x_img = j_index * tile_width_img

        # Place the tile into the full image
        result_tensor[start_y:start_y + tile_height, start_x:start_x + tile_width] = tensor
        result_image[start_y_img:start_y_img + tile_height_img, start_x_img:start_x_img + tile_width_img] = image

    return result_tensor, result_image
