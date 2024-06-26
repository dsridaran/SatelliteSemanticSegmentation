import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from scipy.spatial import distance
import os
import csv

colors = {
    'Urban': np.array([170, 126, 63]),
    'Tree': np.array([42, 164, 48]),
    'Cloud': np.array([255, 255, 255]),
    'Water': np.array([0, 0, 255]),
    'None': np.array([0, 0, 0])
}

def try_sam(model, image_path, prompt, box_threshold, text_threshold, height, width):
    """
    Generate SAM predictions or return empty mask if no objects detected.

    Parameters:
        model (Model): SAM model.
        image_path (str): Path to the image file on which predictions are to be made.
        prompt (str): The prompt based on which the model generates predictions.
        box_threshold (float): The threshold value for determining significant boxes in model prediction.
        text_threshold (float): The threshold value for filtering significant text in model predictions.
        height (int): The height of the tensor to be returned in case of an error.
        width (int): The width of the tensor to be returned in case of an error.

    Returns:
        Tensor with logit values.
    """
    try:
        masks, boxes, phrases, logits = model.predict(image_path, prompt, box_threshold = box_threshold, text_threshold = text_threshold, return_results = True)
    except Exception as e:
        masks = torch.zeros((1, height, width), dtype = torch.bool)
        logits = torch.tensor([0])
    logits[logits < text_threshold] = 0
    return get_logits(masks, logits)

def get_logits(masks, logits):
    """
    Return tensor with average logits per pixel from SAM output.

    Parameters:
        masks (torch.Tensor): A tensor representing binary masks for the logits. 
        logits (torch.Tensor): A tensor containing logits. This tensor should have a length that matches the number of masks.

    Returns:
        torch.Tensor: A tensor with average logits per pixel.
    """
    if logits.shape == torch.Size([1]):
        # Process for a single mask-logit pair
        modified_tensor = torch.where(masks, logits, torch.tensor(0.0))
        modified_tensor = modified_tensor.squeeze()
    else:
        # Process for multiple mask-logit pairs
        modified_tensor = torch.zeros_like(logits.unsqueeze(1).unsqueeze(2).expand(-1, masks.shape[1], masks.shape[2]))
        for i in range(masks.shape[0]):
            modified_tensor[i] = torch.where(
                masks[i], 
                logits.unsqueeze(1).unsqueeze(2).expand(-1, masks.shape[1], masks.shape[2])[i], 
                torch.zeros_like(logits[i])
            )
        modified_tensor = torch.mean(modified_tensor, dim = 0)
    return modified_tensor

def determine_most_likely(tensors, outcomes, color_map = colors):
    """
    Creates RGB representation of most likely class for each pixel based on the provided class tensors.

    Parameters:
        urban (torch.Tensor): Tensor representing the likelihood of urban areas.
        cloud (torch.Tensor): Tensor representing the likelihood of cloud cover.
        tree (torch.Tensor): Tensor representing the likelihood of tree cover.
        water (torch.Tensor): Tensor representing the likelihood of water bodies.
        color_map (list): Color mapping for features.

    Returns:
        tuple: A tuple containing:
            - result (np.ndarray): An array of the most likely class for each pixel.
            - rgb_image (np.ndarray): An RGB image where each class is represented by a specific color.
    """
    # Determine class with highest value for each pixel
    stacked = torch.stack(tensors, dim = -1)
    result_indices = torch.argmax(stacked, dim = -1)

   # Create result array with labels corresponding to index of highest value
    result = np.empty(result_indices.shape, dtype = object)
    for i, outcome in enumerate(outcomes):
        result[result_indices == i] = outcome

    # Assign "None" to pixels where all probabilities are zero
    none_mask = (stacked == 0).all(dim = -1)
    result[none_mask] = "None"

    # Convert class labels to RGB image based on color map
    rgb_image = np.array([color_map[item] for row in result for item in row])
    rgb_image = rgb_image.reshape(result.shape[0], result.shape[1], 3)
    rgb_image = rgb_image / 255
    return result, rgb_image

def evaluation_metrics(ground_truth, prediction):
    """
    Calculate various evaluation metrics for image classification.

    Parameters:
        ground_truth (np.array): The ground truth image data as an array.
        prediction (np.array): The predicted labels as an array.

    Returns:
        tuple: A tuple containing:
            - confusion matrix (normalized)
            - accuracy
            - weighted F1 score
            - Dice scores for each class
            - Count of true and predicted labels for each class
    """
    labels = ["Urban", "Cloud", "Tree", "Water"]
    y_true = get_ground_truth(ground_truth).flatten()
    y_pred = prediction.flatten()

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average = 'weighted')
    dice_scores = dice_score_multiclass(y_true, y_pred, labels)

    # Compute actual and predicted counts by class
    counts = {f"y_true_{label}": np.count_nonzero(y_true == label) for label in labels}
    counts.update({f"y_pred_{label}": np.count_nonzero(y_pred == label) for label in labels})

    return accuracy, weighted_f1, dice_scores, counts
    
def get_ground_truth(groud_truth, color_map = colors):
    """
    Convert an image array into a classification map based on predefined color mappings.

    Parameters:
        ground_truth (np.array): The ground truth image data in RGB format.
        color_map (list): Color mapping for features.

    Returns:
        np.array: An array of shape (H, W) with classified labels.
    """    
    image_array = np.array(groud_truth)
    if image_array.shape[-1] > 3:
        image_array = image_array[:, :, :3]
    colors = np.array(list(color_map.values()))
    labels = list(color_map.keys())
    pixels = image_array.reshape(-1, 3)
    dist = distance.cdist(pixels, colors, 'euclidean')
    nearest_color_indices = np.argmin(dist, axis = 1)
    classified_pixels = np.array([labels[idx] for idx in nearest_color_indices])
    return classified_pixels.reshape(image_array.shape[0], image_array.shape[1])

def dice_score_multiclass(y_true, y_pred, labels):
    """
    Compute the Dice score for each class in a multiclass setting.

    Parameters:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted labels.
        labels (list): List of labels.

    Returns:
        list: Dice scores for each class.
    """
    dice_scores = []
    for i in labels:
        true_binary = (y_true == i)
        pred_binary = (y_pred == i)
        intersection = np.logical_and(true_binary, pred_binary)
        union = np.logical_or(true_binary, pred_binary)
        
        if union.sum() == 0:
            dice = 1.0 if intersection.sum() == 0 else 0.0
        else:
            dice = 2.0 * intersection.sum() / (true_binary.sum() + pred_binary.sum())
        
        dice_scores.append(dice)
    return dice_scores
    
def append_results_to_csv(file_path, folder, image, tt, bt, urban_prompt, cloud_prompt, tree_prompt, water_prompt, accuracy, weighted_f1, dice_scores, counts):
    """
    Append results of model evaluation to CSV file.

    Parameters:
        file_path (str): Path to CSV file where results will be stored.
        folder (str): Folder identifier for outputs.
        image (str): Identifier for image used.
        tt (float): Text threshold parameter used.
        bt (float): Box threshold parameter used.
        urban_prompt, cloud_prompt, tree_prompt, water_prompt (str): Prompts used for each classification category.
        accuracy (float): Accuracy score.
        weighted_f1 (float): Weighted F1 score.
        dice_scores (list): List of Dice scores for each class.
        counts (dict): Dictionary containing counts of actual and predicted labels per class.
    """
    # Create file if required
    if not os.path.isfile(file_path):
        header = [
            # Parameters
            'image', 'folder', 'text_threshold', 'box_threshold', 'urban_prompt', 'cloud_prompt', 'tree_prompt', 'water_prompt', 
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
    data_to_append = [image, folder, tt, bt, urban_prompt, cloud_prompt, tree_prompt, water_prompt, accuracy, weighted_f1, *dice_scores_flat, *counts_ordered]

    # Append to the CSV file
    with open(file_path, 'a', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(data_to_append)