import os
import csv
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from plot_tools import plot_all_tensors, plot_full_result
from model_utils import try_sam, determine_most_likely, evaluation_metrics

def train_ensemble_model(sam, image, urban_prompt, cloud_prompt, tree_prompt, water_prompt, bt, tt, save_images = False, save_results = False):
    """
    Train ensemble model and evaluate it against a ground truth, saving the results and images.
    
    Parameters:
        sam (model): SAM model.
        image (str): Identifier for the image (without file extension).
        urban_prompt, cloud_prompt, tree_prompt, water_prompt (str): Prompts for different classes.
        bt (float): Box threshold.
        tt (float): Text threshold.
        save_images (bool): Saves results images if True.
        save_results (bool): Saves model results if True.
    """
    # Paths for raw and ground truth images
    raw = f'../data/gamma/{image}.png'
    ground_truth = f'../data/labeled/{image}_colored.png'
    raw_img = Image.open(raw)
    width, height = raw_img.size

    # Generate SAM masks
    tensor_urban = try_sam(sam, raw, urban_prompt, bt, tt, height, width)
    tensor_cloud = try_sam(sam, raw, cloud_prompt, bt, tt, height, width)
    tensor_tree = try_sam(sam, raw, tree_prompt, bt, tt, height, width)
    tensor_water = try_sam(sam, raw, water_prompt, bt, tt, height, width)

    # Create combined mask
    result_tensor, prediction = determine_most_likely([tensor_urban, tensor_cloud, tensor_tree, tensor_water], ["Urban", "Cloud", "Tree", "Water"])
    
    # Extract performance metrics
    ground_truth_img = Image.open(ground_truth)
    cm, accuracy, weighted_f1, dice_scores, counts = evaluation_metrics(ground_truth_img, result_tensor)

    # Save plots
    if save_images:
        folder_path = f'../results/ensemble/{image}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Plot individual tensors
        filename = f'../results/ensemble/{image}/indiv_prompt_{urban_prompt}_{cloud_prompt}_{tree_prompt}_{water_prompt}_bt{bt}_tt{tt}.png'
        plot_all_tensors(tensor_urban, tensor_cloud, tensor_tree, tensor_water, urban_prompt, cloud_prompt, tree_prompt, water_prompt, filename)
        
        # Plot raw, ground truth, and prediction
        filename = f'../results/ensemble/{image}/combined_prompt_{urban_prompt}_{cloud_prompt}_{tree_prompt}_{water_prompt}_bt{bt}_tt{tt}.png'
        plot_full_result(raw, ground_truth, prediction, filename)

    # Create output folder for storing result images
    if save_results:
        file_path = '../results/ensemble_results.csv'
        append_results_to_csv(file_path, image, tt, bt, urban_prompt, cloud_prompt, tree_prompt, water_prompt, cm, accuracy, weighted_f1, dice_scores, counts)

def append_results_to_csv(file_path, image, tt, bt, urban_prompt, cloud_prompt, tree_prompt, water_prompt, cm, accuracy, weighted_f1, dice_scores, counts):
    """
    Append results of model evaluation to CSV file.

    Parameters:
        file_path (str): Path to CSV file where results will be stored.
        image (str): Identifier for image used.
        tt (float): Text threshold parameter used.
        bt (float): Box threshold parameter used.
        urban_prompt, cloud_prompt, tree_prompt, water_prompt (str): Prompts used for each classification category.
        cm (np.array): Confusion matrix.
        accuracy (float): Accuracy score.
        weighted_f1 (float): Weighted F1 score.
        dice_scores (list): List of Dice scores for each class.
        counts (dict): Dictionary containing counts of actual and predicted labels per class.
    """
    # Create file if required
    if not os.path.isfile(file_path):
        header = [
            # Parameters
            'image', 'text_threshold', 'box_threshold', 'urban_prompt', 'cloud_prompt', 'tree_prompt', 'water_prompt', 
            # Evaluation metrics
            'accuracy', 'weighted_f1', 'urban_dice', 'cloud_dice', 'tree_dice', 'water_dice', 
            # Pixel summary
            'y_true_urban', 'y_true_cloud', 'y_true_tree', 'y_true_water', 'y_pred_urban', 'y_pred_cloud', 'y_pred_tree', 'y_pred_water'
        ]
        with open(file_path, 'w', newline = '') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    # Flatten dice scores and extract counts
    dice_scores_flat = [dice_scores[i] for i in range(len(dice_scores))]
    counts_ordered = [counts[f"y_true_{label}"] for label in ["Urban", "Cloud", "Tree", "Water"]] + [counts[f"y_pred_{label}"] for label in ["Urban", "Cloud", "Tree", "Water"]]

    # Prepare single row of data to append
    data_to_append = [image, tt, bt, urban_prompt, cloud_prompt, tree_prompt, water_prompt, accuracy, weighted_f1, *dice_scores_flat, *counts_ordered]

    # Append to the CSV file
    with open(file_path, 'a', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(data_to_append)