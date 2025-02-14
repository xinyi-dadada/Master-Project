import os
import matplotlib.pyplot as plt
from PIL import Image


def merge_figures(folder_path, rows=14, cols=3):
    # Get all image file paths from the folder, sorted to maintain order
    image_paths = sorted(
        [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))])

    # Create a large figure with specified number of rows and columns
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    fig.suptitle("Merged Figures", fontsize=16)

    # Flatten the axes array to iterate over them easily
    axes = axes.flatten()

    # Plot each image in a separate subplot
    for idx, img_path in enumerate(image_paths):
        if idx < len(axes):  # Check to avoid index errors if there are fewer images than subplots
            img = Image.open(img_path)  # Open the image
            axes[idx].imshow(img)  # Display the image on the subplot
            axes[idx].axis('off')  # Hide the axes for cleaner display
            axes[idx].set_title(f"Figure {idx + 1}", fontsize=10)  # Optional: add title

    # Turn off any unused subplots
    for j in range(len(image_paths), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust space to include the title
    plt.savefig('/home/Shared/xinyi/blob1/thesis/figure/result_pca/all_result_pca.png')


# Use the function by providing the folder containing your images
folder_path = '/home/Shared/xinyi/blob1/thesis/figure/result_pca'
merge_figures(folder_path, rows=14, cols=3)