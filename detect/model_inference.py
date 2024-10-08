import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from skimage import measure
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

# Define the same model architecture as the one used during training
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
       
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, out_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def load_model(model_path):
    model = UNet3D(in_channels=2, out_channels=1)  
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load the saved weights
    model.eval()  
    return model


def predict(model, nii_file_path):
    image = nib.load(nii_file_path).get_fdata()  
    image = torch.tensor(image, dtype=torch.float32)
    image = image.permute(3, 0, 1, 2) 
    image = image.unsqueeze(0)  

    # Forward pass through the model
    with torch.no_grad():
        output = model(image)  # Get predictions

    # Post-process output (assuming binary output)
    predicted_mask = output.squeeze(0).numpy()  # Remove batch dimension
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  # Apply threshold to get binary mask

    return predicted_mask

import zipfile
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os

def save_mask_as_image(mask, output_image_path_template):
    """
    Save each slice of the 3D mask as a separate PNG image.
    
    Parameters:
    mask: 3D numpy array (H, W, D)
    output_image_path_template: String template for output file paths (e.g., 'slice_{}.png')

    Returns:
    slice_image_paths: List of file paths to the saved images
    """
    # Ensure the mask is at least 3D (in case it's 4D, e.g., (1, H, W, D))
    mask = np.squeeze(mask)  # Remove any singleton dimensions, like the first dimension with size 1

    num_slices = mask.shape[2]  # Assuming the mask now has shape (H, W, D)
    slice_image_paths = []  # Initialize a list to store image paths

    for i in range(num_slices):
        # Format the output image path for the current slice
        output_image_path = output_image_path_template.format(i)

        # Create an empty placeholder file in default storage
        temp_image_file = default_storage.save(output_image_path, ContentFile(b''))
        slice_image_paths.append(temp_image_file)

        # Plot the slice and save it to the path
        plt.imshow(mask[:, :, i], cmap='gray')
        plt.axis('off')  # Hide axes
        plt.savefig(default_storage.path(temp_image_file), bbox_inches='tight', pad_inches=0)
        plt.close()

    return slice_image_paths  # Return the list of saved image paths


def create_zip_with_predictions(output_nii_path, slice_image_paths):
    """
    Creates a ZIP file containing the NIfTI file and all the slice images.
    
    Parameters:
    output_nii_path: Path to the NIfTI prediction file
    slice_image_paths: List of paths to the saved slice images
    """
    # Create a ZIP file in the default storage
    zip_filename = default_storage.save('predictions.zip', ContentFile(b''))

    with zipfile.ZipFile(default_storage.path(zip_filename), 'w') as zipf:
        # Add the NIfTI file to the ZIP
        zipf.write(default_storage.path(output_nii_path), arcname='prediction_output.nii')

        # Add all the slice images to the ZIP
        for slice_image_path in slice_image_paths:
            zipf.write(default_storage.path(slice_image_path), arcname=os.path.basename(slice_image_path))
    
    return zip_filename


def save_mask_as_image_with_circles(mask, output_image_path_template):
    """
    Save each slice of the 3D mask as a separate PNG image with cancer regions circled in red.
    
    Parameters:
    mask: 3D numpy array (H, W, D)
    output_image_path_template: String template for output file paths (e.g., 'slice_{}.png')

    Returns:
    slice_image_paths: List of file paths to the saved images
    """
    mask = np.squeeze(mask)  # Ensure the mask is at least 3D (e.g., (H, W, D))
    num_slices = mask.shape[2]
    slice_image_paths = []

    for i in range(num_slices):
        output_image_path = output_image_path_template.format(i)
        temp_image_file = default_storage.save(output_image_path, ContentFile(b''))
        slice_image_paths.append(temp_image_file)

        # Detect cancerous regions (non-zero areas in the mask)
        cancerous_regions = mask[:, :, i] > 0.5

        # Plot the mask slice
        plt.imshow(mask[:, :, i], cmap='gray')

        # Draw red circles around cancerous regions
        contours = measure.find_contours(cancerous_regions, 0.5)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')

        # Save the plot as an image
        plt.axis('off')  # Hide axes
        plt.savefig(default_storage.path(temp_image_file), bbox_inches='tight', pad_inches=0)
        plt.close()

    return slice_image_paths