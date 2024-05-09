import os
from PIL import Image
from plot_tools import plot_full_result
from model_utils import try_sam, determine_most_likely, evaluation_metrics, append_results_to_csv

def train_individual_model(sam, image, folder, prompt, bt, tt, object_being_predicted, save_images = False, save_results = False):
    """
    Train individual model and evaluate it against a ground truth, saving the results and images.
    
    Parameters:
        sam (model): SAM model.
        image (str): Identifier for the image (without file extension).
        folder (str): Identifier of image type ('rgb' or 'gamma').
        prompt (str): Prompts for segmentation.
        bt (float): Box threshold.
        tt (float): Text threshold.
        object_being_predicted (str): Feature being segmented ('Urban', 'Tree', 'Cloud', or 'Water').
        save_images (bool): Saves results images if True.
        save_results (bool): Saves model results if True.
    """
    # Paths for raw and ground truth images
    raw = f'../data/{folder}/{image}.png'
    ground_truth = f'../data/labeled/{image}_colored.png'
    raw_img = Image.open(raw)
    width, height = raw_img.size

    # Check valid response
    urban_prompt = ""; cloud_prompt = ""; tree_prompt = ""; water_prompt = "";
    if object_being_predicted == "Urban":
        urban_prompt = prompt
    elif object_being_predicted == "Water":
        water_prompt = prompt
    elif object_being_predicted == "Tree":
        tree_prompt = prompt
    elif object_being_predicted == "Cloud":
        cloud_prompt = prompt
    else:
        print("object_being_predicted must be 'Urban', 'Water', 'Tree', or 'Cloud'.")
        return

    # Generate SAM masks
    tensor = try_sam(sam, raw, prompt, bt, tt, height, width)
    result_tensor, prediction = determine_most_likely([tensor], [object_being_predicted])

    # Extract performance metrics
    ground_truth_img = Image.open(ground_truth)
    accuracy, weighted_f1, dice_scores, counts = evaluation_metrics(ground_truth_img, result_tensor)
    
    # Save results
    if save_results:
        file_path = '../results/individual_results.csv'
        append_results_to_csv(file_path, folder, image, tt, bt, urban_prompt, cloud_prompt, tree_prompt, water_prompt, accuracy, weighted_f1, dice_scores, counts)

    # Save plots
    if save_images:
        folder_path = f'../results/individual/{folder}/{image}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    filename = f'../results/individual/{folder}/{image}/individual_{folder}_{prompt}_{object_being_predicted}_bt{bt}_tt{tt}.png'
    return plot_full_result(raw, ground_truth, prediction, filename, save_images)