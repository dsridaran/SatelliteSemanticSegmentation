import matplotlib.pyplot as plt

def plot_all_tensors(tensor_urban, tensor_cloud, tensor_tree, tensor_water, urban_prompt, cloud_prompt, tree_prompt, water_prompt, filename):
    """
    Plots four tensors in a single figure with individual color maps and titles based on specified prompts.
    
    Parameters:
        tensor_urban, tensor_cloud, tensor_tree, tensor_water (torch.Tensor): Tensors to be plotted.
        urban_prompt, cloud_prompt, tree_prompt, water_prompt (str): Prompts used to generate the respective tensors.
        filename (str): File path where the resulting plot will be saved.
    """
    fig, axes = plt.subplots(1, 4, figsize = (18, 6), constrained_layout = True)
    
    # Plot urban tensor
    im1 = plot_individual_tensor(axes[0], tensor_urban, "Reds")
    plt.colorbar(im1, ax = axes[0]); axes[0].set_title(f'Prompt: {urban_prompt}')
    
    # Plot cloud tensor
    im2 = plot_individual_tensor(axes[1], tensor_cloud, "Purples")
    plt.colorbar(im2, ax = axes[1]); axes[1].set_title(f'Prompt: {cloud_prompt}')
    
    # Plot tree tensor
    im3 = plot_individual_tensor(axes[2], tensor_tree, "Greens")
    plt.colorbar(im3, ax = axes[2]); axes[2].set_title(f'Prompt: {tree_prompt}')
    
    # Plot water tensor
    im4 = plot_individual_tensor(axes[3], tensor_water, "Blues")
    plt.colorbar(im4, ax = axes[3]); axes[3].set_title(f'Prompt: {water_prompt}')
    
    fig.savefig(filename)
    plt.clf()
    plt.close(fig) 
    
def plot_full_result(raw, ground_truth, prediction, filename):
    """
    Plots a comparison of a raw image, its manually labeled ground truth, and a model's prediction in a single figure.

    Parameters:
        raw (str): File path to the raw image.
        ground_truth (str): File path to the image of manually labeled ground truth.
        prediction (np.array): The prediction image data.
        filename (str): File path where the resulting plot will be saved.
    """
    fig, axes = plt.subplots(1, 3, figsize = (18, 6), constrained_layout = True)
    fig.subplots_adjust(top = 0.9)
    
    # Plot raw image
    img = mpimg.imread(raw)
    axes[0].imshow(img); axes[0].axis('off'); axes[0].set_title('Raw Image');

    # Plot ground truth
    ground_truth_img = mpimg.imread(ground_truth)
    axes[1].imshow(ground_truth_img); axes[1].axis('off'); axes[1].set_title('Manually Labeled Image')

    # Plot prediction
    axes[2].imshow(prediction); axes[2].axis('off'); axes[2].set_title('Ensemble Prediction')
    
    fig.savefig(filename)
    plt.clf()
    plt.close(fig)  

def plot_individual_tensor(ax, tensor, color):
    """
    Plot a single tensor on a given axis with specified color mapping.

    Parameters:
        ax (matplotlib.axes.Axes): The axes object on which the tensor will be plotted.
        tensor (array-like): The tensor data to plot.
        color (str): The colormap string to apply to the tensor visualization.

    Returns:
        matplotlib.image.AxesImage: The image object created by imshow used for further tweaking or legend addition.
    """
    im = ax.imshow(tensor, cmap = color, vmin = 0, vmax = 1)
    ax.axis('off')
    return im

def plot_overlayed_tensors(tensors, colors, title):
    """
    Plot multiple tensors overlayed on each other with specified colors and alpha blending.

    Parameters:
        tensors (list of array-like): A list of tensors to be overlayed and plotted.
        colors (list of str): A list of colormap strings corresponding to each tensor.
        title (str): The title of the plot.

    Details:
        This function creates an overlay of multiple tensors each with a specified colormap. 
        It uses semi-transparent overlays to allow visualization of overlapping areas.
    """
    plt.figure(figsize = (6, 6))
    vmax = max(tensor.max().item() for tensor in tensors)
    for tensor, color in zip(tensors, colors):
        plt.imshow(tensor, cmap = color, vmin = 0, vmax = vmax, alpha = 0.5)
    plt.colorbar()
    plt.title(title)
    plt.show()
    
def add_grid_lines(image, num_tiles = 5):
    """
    Adds grid lines to an image.

    Parameters:
        image (np.array): A NumPy array representing an image. The array should have dimensions (height, width, channels).
        num_tiles (int, optional): The number of tiles per row and column. Default value is 5.

    Returns:
        np.array: The modified image with grid lines.
    """
    height, width, channels = image.shape
    tile_height = height // num_tiles
    tile_width = width // num_tiles
    for i in range(1, num_tiles):
        image[:, i * tile_width - 1:i * tile_width + 1, :] = 0
        image[i * tile_height - 1:i * tile_height + 1, :, :] = 0
    return image