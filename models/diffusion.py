import torch
import logging

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
            for i in reversed(range(1, self.noise_steps)):
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
