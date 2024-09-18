import os
import copy
import numpy as np
from tqdm import tqdm
import random
from PIL import Image
import torch
import torch.nn as nn
from torch import optim
import torchvision
import logging
from torch.utils.data import DataLoader, Dataset, random_split
import math
from torchvision import models
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import transforms
import zipfile
import nrrd
from io import BytesIO
import torch.nn.utils.spectral_norm as spectral_norm
import torchio as tio
from torchmetrics.image import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import nrrd


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        """
        Update the moving average of the model parameters.
        Args:
            ma_model: The model with moving average parameters.
            current_model: The current model being trained.
        """
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """
        Compute the updated average using exponential moving average formula.
        Args:
            old: Old parameter value.
            new: New parameter value.
        Returns:
            Updated parameter value.
        """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=1000):
        """
        Perform a step of EMA update.
        Args:
            ema_model: The model with moving average parameters.
            model: The current model being trained.
            step_start_ema: The step at which to start EMA updates.
        """
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        """
        Reset the EMA model parameters to the current model parameters.
        Args:
            ema_model: The model with moving average parameters.
            model: The current model being trained.
        """
        ema_model.load_state_dict(model.state_dict())

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Forward pass for self-attention mechanism.
        Args:
            x: Input tensor.
        Returns:
            Tensor after applying self-attention.
        """
        batch_size, channels, height, width = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False, with_attention=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = spectral_norm(nn.Conv2d(2*in_ch, out_ch, 3, padding=1))
            self.transform = spectral_norm(nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1))
        else:
            self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            self.transform = spectral_norm(nn.Conv2d(out_ch, out_ch, 4, 2, 1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        self.selu  = nn.SiLU()
        self.with_attention = with_attention
        if self.with_attention:
            self.attention = SelfAttention(out_ch)

    def forward(self, x, t):
        """
        Forward pass for the block.
        Args:
            x: Input tensor.
            t: Time embedding tensor.
        Returns:
            Transformed tensor after applying convolution, time embedding, and optional attention.
        """
        # First Conv
        h = self.relu(self.bnorm1(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Apply attention if required
        if self.with_attention:
            h = self.attention(h)
        # Second Conv
        h = self.relu(self.bnorm2(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Generate sinusoidal position embeddings for a given time step.
        Args:
            time: Input tensor representing the time step.
        Returns:
            Tensor containing sinusoidal position embeddings.
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SimpleUnet(nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        image_channels = 1
        cond_channels = 1
        total_channels = image_channels + cond_channels
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1
        time_emb_dim = 256

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_emb_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim)
        )

        self.conv0 = nn.Conv2d(in_channels=total_channels, out_channels=down_channels[0], kernel_size=3, padding=1)

        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1],
                                          time_emb_dim, with_attention=(i % 2 == 0))
                                    for i in range(len(down_channels) - 1)])

        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1],
                                        time_emb_dim, up=True, with_attention=True)
                                  for i in range(len(up_channels) - 1)])
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, cond, timestep, y=None):
        t = self.time_mlp(timestep)
        if y is not None:
            t += self.label_emb(y)

        # Ensure x and cond have the same dimensions
        assert x.shape[2:] == cond.shape[2:], "Input and conditioning images must have the same spatial dimensions."

        x = torch.cat((x, cond), dim=1)
        x = self.conv0(x)

        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.01, img_size=160, type="unconditional", device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.type = type

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps, device=self.device)

    def noise_images(self, x, t, noise):
        sqrt_alpha_hat = self.sqrt_alpha_hat[t][:, None, None, None]
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t][:, None, None, None]
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=0, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels=None, cond_images=None, cfg_scale=8):
        if self.type != "unconditional" and labels is None:
            raise ValueError('Labels must be passed to perform conditional sampling.')
        if self.type != "unconditional" and cond_images is None:
            raise ValueError('Conditional images must be passed to perform conditional sampling.')

        logging.info(f"Sampling {n} new images....")
        model.eval()

        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)  # Initialize the noise
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)

                if self.type == "unconditional":
                    predicted_noise = model(x, t)
                else:
                    unconditional_noise = model(x, cond_images, t, None)
                    conditional_noise = model(x, cond_images, t, labels)
                    predicted_noise = (1 + cfg_scale) * conditional_noise - cfg_scale * unconditional_noise

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                # Sample noise for each timestep
                current_noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                x = (x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha) + torch.sqrt(beta) * current_noise

        model.train()
        return x


class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[0, 5, 10, 19, 28]):
        """
        Initializes the PerceptualLoss class using VGG19 layers to compute the perceptual loss.
        """
        super(PerceptualLoss, self).__init__()
        
        # Load VGG19 model
        vgg = models.vgg19(weights=None)  # Load VGG19 without pretrained weights
        
        # Load custom weights
        state_dict = torch.load("/home/hpc/iwi5/iwi5199h/vgg19-dcbb9e9d.pth")  
        
        # Modify the first convolutional layer to accept 1 channel instead of 3
        vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        
        # Load the modified state_dict while ignoring the weights for the modified layer
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('features.0')}
        vgg.load_state_dict(filtered_state_dict, strict=False)
        
        self.layers = nn.ModuleList([vgg.features[i] for i in feature_layers])
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        """
        Computes the perceptual loss between the input images x and y.
        """
        loss = 0.0
        for layer in self.layers:
            x = layer(x)
            y = layer(y)
            loss += F.mse_loss(x, y)
        return loss



label_name = ['orignal', '40kev', '90kev', '140kev']

def save_images(images, labels, path, nrow=4, figsize=(10, 10)):
    """
    Saves a batch of images with labels above each image using matplotlib.
    Args:
        images: Tensor of shape (N, C, H, W) - a batch of images
        labels: List or tensor of labels corresponding to each image
        path: Path where the image grid will be saved
        nrow: Number of images per row
        figsize: Size of the figure for matplotlib
    """
    # Check if the path contains a directory
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    # Convert images from Tensor to numpy array
    images = images.permute(0, 2, 3, 1).to('cpu').numpy()  # Shape (N, H, W, C)
    
    # Normalize and convert to [0, 1] range
    images = (images - images.min()) / (images.max() - images.min() + 1e-5)
    
    images = images * 255
    
    num_images = images.shape[0]
    
    # Determine number of rows based on nrow
    ncol = nrow
    nrows = int(np.ceil(num_images / nrow))

    fig, axes = plt.subplots(nrows, ncol, figsize=figsize)
    
    # If there's only one row or column, we make sure axes is 2D
    if nrows == 1:
        axes = np.expand_dims(axes, 0)
    if ncol == 1:
        axes = np.expand_dims(axes, 1)
    
    # Plot each image and label
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i],cmap='gray')  # Plot the image
            ax.set_title(str(label_name[i]), fontsize=12)  # Set the label as title using label_name
            ax.axis('off')  # Turn off axis
        else:
            ax.axis('off')  # Turn off axis for extra subplots if images are fewer
    
    # Adjust layout so labels don't overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(path)
    plt.close()



    

class NRRDSliceDataset(Dataset):
    def __init__(self, input_slices, target_slices, labels, transform=None):
        self.input_slices = input_slices
        self.target_slices = target_slices
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.input_slices)

    def __getitem__(self, idx):
        input_slice = self.input_slices[idx]
        target_slice = self.target_slices[idx]
        label = self.labels[idx]

        input_slice = torch.tensor(input_slice, dtype=torch.float32).unsqueeze(0)
        target_slice = torch.tensor(target_slice, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            input_slice = self.transform(input_slice)
            target_slice = self.transform(target_slice)

        input_slice = (input_slice - input_slice.min()) / (input_slice.max() - input_slice.min())
        input_slice = input_slice * 2 - 1

        target_slice = (target_slice - target_slice.min()) / (target_slice.max() - target_slice.min())
        target_slice = target_slice * 2 - 1

        return input_slice, target_slice, torch.tensor(label, dtype=torch.long)

def load_nrrd_slices_from_zip(zip_file, file_name):
    with zip_file.open(file_name) as file:
        data = BytesIO(file.read())
        data.seek(0)
        magic_line = data.read(4)
        print(f"Reading file: {file_name}")
        print(f"Magic line: {magic_line}")
        if magic_line != b'NRRD':
            print(f"Invalid NRRD magic line: {magic_line}")
            print(f"First few bytes: {data.getvalue()[:10]}")
            return []  # Return an empty list to skip invalid files
        data.seek(0)
        header = nrrd.read_header(data)
        data.seek(0)
        volume = nrrd.read_data(header, data)
        slices = [volume[:, :, i] for i in range(40, volume.shape[2] - 40)]
        return slices

# Paths to the zip files
zip_files = ['24040515_chicken.zip', '24041710_chicken.zip', '24052912_chicken.zip', '24053112_chicken.zip']

input_slices = []
target_slices = []
labels = []


label_mapping = {'40kev': 0, '90kev': 1, '140kev': 2}

# Extract and process the files from the zip archives
for zip_file_path in zip_files:
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        for file_name in zip_file.namelist():
            print(f"Found file in zip: {file_name}")
            if file_name.endswith('.nrrd'):
                if 'spp_me70' in file_name:
                    slices = load_nrrd_slices_from_zip(zip_file, file_name)
                    if slices:
                        input_slices.extend(slices)
                elif any(kev in file_name for kev in label_mapping.keys()):
                    slices = load_nrrd_slices_from_zip(zip_file, file_name)
                    if slices:
                        target_slices.extend(slices)
                        label_key = [kev for kev in label_mapping.keys() if kev in file_name][0]
                        labels.extend([label_mapping[label_key]] * len(slices))

# Ensure input slices are repeated to match the target slices
input_slices = input_slices * (len(target_slices) // len(input_slices))

transformations = transforms.Compose([
    transforms.Resize((160, 160), transforms.InterpolationMode.BILINEAR)
])

dataset = NRRDSliceDataset(input_slices=input_slices, target_slices=target_slices, labels=labels, transform=transformations)

# Calculate the number of samples for testing (0.01% of the dataset)
test_size = int(0.01 * len(dataset))
train_size = len(dataset) - test_size

# Split the dataset into training and testing sets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoader objects for both training and testing sets
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

def test_model(model, dataloader, diffusion, ssim_loss, device='cuda'):
    model.eval()
    total_ssim = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for i, (input_slice, target_slice, label) in enumerate(dataloader):
            input_slice = input_slice.to(device)
            target_slice = target_slice.to(device)
            label = label.to(device)

            t = diffusion.sample_timesteps(input_slice.shape[0]).to(device)
            noise = torch.randn_like(input_slice)
            x_t, noise = diffusion.noise_images(input_slice, t, noise)

            predicted_noise = model(x_t, input_slice, t, label)

            ssim_val = ssim_loss(noise, predicted_noise)
            total_ssim += ssim_val.item()
    
    avg_ssim = total_ssim / num_batches
    print(f"Average SSIM Score on test set: {avg_ssim:.5f}")
    return avg_ssim



def train(device='cuda', sampling_type='unconditional', num_classes=3, lr=0.002, image_size=160, epochs=200):
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
        pbar = tqdm(train_dataloader)
        epoch_loss = 0
        epoch_ssim_loss = 0
    
        for i, (input_slice, target_slice, label) in enumerate(pbar):
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
    
        # Test the model on the test set every epoch
        if epoch % 5 == 0:  # You can adjust this to test at different intervals
            test_ssim_score = test_model(model, test_dataloader, diffusion, ssim_loss, device=device)
            logging.info(f"Test SSIM Score at Epoch {epoch}: {test_ssim_score:.5f}")
    
        if epoch % 10 == 0 and epoch != 0:
            with torch.no_grad():
                
                if sampling_type != "unconditional":
                    labels = torch.arange(num_classes).long().to(device)
                    sampled_images = diffusion.sample(model, n=len(labels), labels=labels, cond_images=input_slice.repeat(len(labels), 1, 1, 1))
                    images = torch.cat((input_slice, sampled_images),dim=0)
                    save_images(images, labels, os.path.join("results", f"conditional_{epoch}.png"))
                else:
                    sampled_images = diffusion.sample(model, n=input_slice.shape[0], cond_images=input_slice)
                    save_images(sampled_images, os.path.join("results", f"unconditional_{epoch}.png"))


                
def evaluate_and_generate_images(model, dataloader, diffusion, ssim_loss, device, sampling_type, image_save_dir="test_results"):
    model.eval()
    epoch_ssim_loss = 0
    generated_images = []

    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    with torch.no_grad():
        for i, (input_slice, target_slice, label) in enumerate(dataloader):
            input_slice = input_slice.to(device)
            target_slice = target_slice.to(device)
            label = label.to(device)

            # Generate images using the model
            if sampling_type == 'unconditional':
                sampled_images = diffusion.sample(model, n=input_slice.shape[0], cond_images=input_slice)
            else:
                sampled_images = diffusion.sample(model, n=1, labels=label, cond_images=input_slice)

            # Save generated images
            save_images(sampled_images, os.path.join(image_save_dir, f"generated_{i}.jpg"))

            # Calculate the SSIM between the generated image and the original target image
            ssim_value = ssim_loss(sampled_images, target_slice).item()
            epoch_ssim_loss += ssim_value

            # Store the generated images and SSIM score
            generated_images.append((sampled_images.cpu(), target_slice.cpu(), ssim_value))

    avg_ssim_loss = epoch_ssim_loss / len(dataloader)
    model.train()

    return avg_ssim_loss, generated_images




train(device='cuda', epochs=501, image_size=160, lr=0.0003, sampling_type='conditional', num_classes=3)
#model = SimpleUnet(num_classes=3).to('cuda')

#model.load_state_dict(torch.load("/home/hpc/iwi5/iwi5199h/results/conditional_100"))
#diffusion = Diffusion(noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=160, type='conditional', device='cuda')
#ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
#evaluate_and_generate_images(model, test_dataloader, diffusion, ssim_loss, device, sampling_type, image_save_dir="test_results")
#for (input_slice, target_slice, label) in test_dataloader:
#    input_slice = input_slice
#    break
#print(input_slice.shape)
#cond_images = input_slice.repeat(3, 1, 1, 1).to("cuda")
#print(cond_images.shape)
#generate_images(sampling_type='conditional',cond_images=cond_images)