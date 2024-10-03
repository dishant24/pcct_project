from helper.dataloader import get_dataloaders
from helper.image_helper import save_images
from helper.losses import PerceptualLoss
from helper.test import evaluate_and_generate_images, generate_image_samples
from models.attention import SelfAttention
from models.diffusion import Diffusion
from models.ema import EMA
from models.unet import SimpleUnet
import os
import copy
import torch
from torch import optim
import torch.nn as nn
import logging
from torchmetrics.image import StructuralSimilarityIndexMeasure

def train(device='cuda', sampling_type='conditional', num_classes=3, lr=0.002, image_size=160, epochs=200, train_dataloader, test_dataloader):
    logging.basicConfig(filename='training.log', level=logging.INFO, 
                        format='%(asctime)s %(levelname)s:%(message)s')
    model = SimpleUnet().to(device) if sampling_type == 'unconditional' else SimpleUnet(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.L1Loss().to(device)
    ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    perceptual_loss = PerceptualLoss().to(device)
    diffusion = Diffusion(noise_steps=1000, beta_start=1e-4, beta_end=0.01, img_size=image_size, type=sampling_type, device=device)

    if sampling_type != "unconditional":
        ema = EMA(0.99)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch}:")
        epoch_loss = 0
        epoch_ssim_loss = 0
    
        for i, (input_slice, target_slice, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_slice = input_slice.to(device)
            target_slice = target_slice.to(device)
            label = label.to(device)
            
            t = diffusion.sample_timesteps(input_slice.shape[0]).to(device)
            noise = torch.randn_like(input_slice)
            x_t, noise = diffusion.noise_images(input_slice, t, noise)
            
            if sampling_type == 'unconditional':
                predicted_noise = model(x_t, input_slice, t)
            else:
                if np.random.random() < 0.10:
                    label = None
                predicted_noise = model(x_t, input_slice, t, label)
    
            mse_loss = mse(noise, predicted_noise)
            perceptual_loss_value = perceptual_loss(noise, predicted_noise)
            ssim_val = ssim_loss(noise, predicted_noise)
            epoch_ssim_loss += ssim_val.item()
    
            # Combine the losses with weighted sum
            loss = 0.7 * mse_loss + 0.2 * perceptual_loss_value + 0.1 * (1 - ssim_val)
            epoch_loss += loss.item()
    
            loss.backward()
            optimizer.step()
    
            if sampling_type != 'unconditional':
                ema.step_ema(ema_model, model)
    
        scheduler.step()
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        avg_epoch_ssim_loss = epoch_ssim_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch} Loss: {avg_epoch_loss:.5f}    SSIM Loss: {avg_epoch_ssim_loss:.5f}")
        
        if epoch % 25 == 0 and epoch != 0:
            with torch.no_grad():
                
                if sampling_type != "unconditional":
                    labels = torch.arange(num_classes).long().to(device)
                    sampled_images = diffusion.sample(model, n=len(labels), labels=labels, cond_images=input_slice.repeat(len(labels), 1, 1, 1))
                    images = torch.cat((input_slice, sampled_images),dim=0)
                    save_images(images, labels, os.path.join("results", f"conditional_{epoch}.png"))
                    torch.save(model , os.path.join("results", f"conditional_{epoch}.h5")
                    evaluate_and_generate_images(model, test_dataloader, diffusion, ssim_loss, device, sampling_type, image_save_dir="test_results")
                else:
                    sampled_images = diffusion.sample(model, n=input_slice.shape[0], cond_images=input_slice)
                    save_images(sampled_images, os.path.join("results", f"unconditional_{epoch}.png"))
        
    
if __name__ == '__main__':
    zip_files = ['24040515_chicken.zip', '24041710_chicken.zip', '24052912_chicken.zip', '24053112_chicken.zip']
    label_mapping = {'40kev': 0, '90kev': 1, '140kev': 2}
    train_dataloader, test_dataloader = get_dataloaders(zip_files, label_mapping, test_split=0.008, batch_size=1)
    train(device='cuda', sampling_type='conditional', num_classes=3, lr=0.0005, image_size=160, epochs=200, train_dataloader, test_dataloader)
    generate_image_samples('/home/hpc/iwi5/iwi5199h/results/conditional_100', test_dataloader, diffusion, ssim_loss, device='cuda', sampling_type ='conditional', image_save_dir="sample_results")