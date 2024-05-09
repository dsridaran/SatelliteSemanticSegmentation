import os
from PIL import Image
from plot_tools import plot_all_tensors, plot_full_result
from model_utils import try_sam, determine_most_likely, evaluation_metrics, append_results_to_csv

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
    accuracy, weighted_f1, dice_scores, counts = evaluation_metrics(ground_truth_img, result_tensor)

    # Save results
    if save_results:
        file_path = '../results/ensemble_results.csv'
        append_results_to_csv(file_path, "ensemble", image, tt, bt, urban_prompt, cloud_prompt, tree_prompt, water_prompt, accuracy, weighted_f1, dice_scores, counts)
        
    # Save plots
    if save_images:
        folder_path = f'../results/ensemble/{image}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Plot individual tensors
    filename = f'../results/ensemble/{image}/indiv_prompt_{urban_prompt}_{cloud_prompt}_{tree_prompt}_{water_prompt}_bt{bt}_tt{tt}.png'
    plot_all_tensors(tensor_urban, tensor_cloud, tensor_tree, tensor_water, urban_prompt, cloud_prompt, tree_prompt, water_prompt, filename, save_images)
    
    # Plot raw, ground truth, and prediction
    filename = f'../results/ensemble/{image}/combined_prompt_{urban_prompt}_{cloud_prompt}_{tree_prompt}_{water_prompt}_bt{bt}_tt{tt}.png'
    return plot_full_result(raw, ground_truth, prediction, filename, save_images)