import os
import matplotlib.pyplot as plt
import numpy as np

def save_images(images, labels, path, nrow=4, figsize=(10, 10)):
    # Create the directory if it doesn't exist
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    # Move images from torch tensor format (B, C, H, W) to (B, H, W, C) and convert to numpy
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    
    # Normalize images to range [0, 255]
    images = (images - images.min()) / (images.max() - images.min() + 1e-5) * 255
    images = images.astype(np.uint8)
    
    # Determine the number of images and adjust grid size dynamically
    num_images = images.shape[0]
    ncol = nrow
    nrows = int(np.ceil(num_images / nrow))
    
    # Create the figure and subplots
    fig, axes = plt.subplots(nrows, ncol, figsize=figsize)
    
    # Flatten the axes array to make indexing easier (handle case with only 1 row or column)
    axes = np.atleast_2d(axes).reshape(-1)

    # Plot each image and corresponding label
    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(images[i], cmap='gray')
            ax.axis('off')
        else:
            # Hide any extra axes that don't have images
            ax.axis('off')
    
    # Adjust layout and save the image
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
