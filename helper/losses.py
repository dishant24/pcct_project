import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure
from helper.image_helper import save_images
import os
        
def evaluate_and_generate_images(model, dataloader, diffusion, ssim_loss, device, sampling_type, image_save_dir="test_results"):
    model.eval()
    epoch_ssim_loss = 0

    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    with torch.no_grad():
        for i, (input_slice, target_slice, label) in enumerate(dataloader):
            input_slice = input_slice.to(device)
            target_slice = target_slice.to(device)
            
            label = label.to(device)

            # Generate images using the model
            if sampling_type == 'unconditional':
                sampled_images = diffusion.sample(model, n=None, cond_images=input_slice)
            else:
                sampled_images = diffusion.sample(model, n=len(label), labels=label, cond_images=input_slice)

            # Save generated images
            save_images(sampled_images, label,  os.path.join(image_save_dir, f"generated_{i}.jpg"))

            # Calculate the SSIM between the generated image and the original target image
            ssim_value = ssim_loss(sampled_images, target_slice).item()
            epoch_ssim_loss += ssim_value


    avg_ssim_loss = epoch_ssim_loss / len(dataloader)

    return avg_ssim_loss
    
    
def generate_image_samples(model_path, dataloader, diffusion, ssim_loss, device='cuda', sampling_type ='conditional', image_save_dir="sample_results"):
    
    model.load_state_dict(torch.load(model_path))
    diffusion = Diffusion(noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=160, type=sampling_type, device='cuda')
    ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    model.eval()
    epoch_ssim_loss = 0

    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    with torch.no_grad():
        for i, (input_slice, target_slice, label) in enumerate(dataloader):
            input_slice = input_slice.to(device)
            label = label.to(device)

            # Generate images using the model
            if sampling_type == 'unconditional':
                sampled_images = diffusion.sample(model, n=None, cond_images=input_slice)
            else:
                sampled_images = diffusion.sample(model, n=len(label), labels=label, cond_images=input_slice)

            # Save generated images
            save_images(sampled_images, label,  os.path.join(image_save_dir, f"generated_{i}.jpg"))

    return avg_ssim_loss