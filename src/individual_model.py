import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from plot_tools import plot_individual_tensor
from model_utils import try_sam, determine_most_likely, evaluation_metrics
import os
import csv

def train_individual_model(sam, image, folder, prompt, bt, tt, object_being_predicted):

    # Create output folder
    folder_path = f'../results/individual/{folder}/{image}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Check valid response
    if object_being_predicted not in ["Urban", "Water", "Tree", "Cloud"]:
        print("object_being_predicted must be 'urban', 'water', 'tree', or 'cloud'.")
        return

    # Create result file
    file_path = '../results/individual_results.csv'
    header = ['image', 'folder', 'text_threshold', 'box_threshold', 'prompt', 'object_being_predicted', 'accuracy', 'weighted_f1', 'dice', 'y_true', 'y_pred']
    if not os.path.isfile(file_path):
        with open(file_path, 'w', newline = '') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            
    # Check if scenario already run
    df = pd.read_csv(file_path)
    condition = (
        (df['image'] == image) 
        & (df['folder'] == folder) 
        & (df['text_threshold'] == tt) 
        & (df['box_threshold'] == bt) 
        & (df['prompt'] == prompt) 
        & (df['object_being_predicted'] == object_being_predicted)
    )
    filtered_df = df[condition]
    
    # Abort if already run
    if len(filtered_df) > 1:
        print("Prompt already run.")
        return 

    # Define image paths
    raw = f'../data/{folder}/{image}.png'
    ground_truth = f'../data/labeled/{image}_colored.png'

    # Extract image dimensions image
    raw_img = Image.open(raw)
    width, height = raw_img.size

    # Generate SAM masks
    tensor = try_sam(sam, raw, prompt, bt, tt, height, width)

    # Plot raw image
    fig, axes = plt.subplots(1, 3, figsize = (18, 6), constrained_layout = True)
    fig.subplots_adjust(top = 0.9)
    img = mpimg.imread(raw)
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title('Raw Image')

    # Plot ground truth
    ground_truth_img = mpimg.imread(ground_truth)
    axes[1].imshow(ground_truth_img)
    axes[1].axis('off')
    axes[1].set_title('Manually Labeled Image')

    # Plot ensemble prediction
    result_tensor, result_image = determine_most_likely([tensor], [object_being_predicted])
    axes[2].imshow(result_image)
    axes[2].axis('off') 
    axes[2].set_title('Individual Prediction')
    fig.savefig(f'../results/individual/{folder}/{image}/individual_{folder}_{prompt}_{object_being_predicted}_bt{bt}_tt{tt}.png')
    plt.clf()
    plt.close(fig)    
    
    # Extract performance metrics
    ground_truth_img = Image.open(ground_truth)
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

    data_to_append = [image, folder, tt, bt, prompt, object_being_predicted, accuracy, weighted_f1, dice, y_true, y_pred]
    with open(file_path, 'a', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(data_to_append)

    return