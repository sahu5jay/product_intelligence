import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerGanConfig:
    trained_model_path_gen = os.path.join("artifacts", "generator.pth")
    trained_model_path_disc = os.path.join("artifacts", "discriminator.pth")
    batch_size = 64
    latent_dim = 100
    epochs = 20
    lr = 0.0002

# 1. Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784), # 28x28 image
            nn.Tanh() # Outputs between -1 and 1
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img

# 2. Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() # Output probability (Real or Fake)
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class ModelTrainerGan:
    def __init__(self):
        self.gan_config = ModelTrainerGanConfig()

    def initiate_gan_training(self):
        try:
            logging.info("Initializing GAN Training for Fashion-MNIST")
            
            # Device configuration
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Data Loading
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            # Using the directory created during Ingestion
            train_data = datasets.FashionMNIST(
                root=os.path.join('artifacts', 'fashion_data'), 
                train=True, 
                download=False, 
                transform=transform
            )
            dataloader = DataLoader(train_data, batch_size=self.gan_config.batch_size, shuffle=True)

            # Initialize Networks
            generator = Generator(self.gan_config.latent_dim).to(device)
            discriminator = Discriminator().to(device)

            # Loss and Optimizers
            adversarial_loss = nn.BCELoss()
            optimizer_G = optim.Adam(generator.parameters(), lr=self.gan_config.lr)
            optimizer_D = optim.Adam(discriminator.parameters(), lr=self.gan_config.lr)

            logging.info("Starting Training Loop...")

            for epoch in range(self.gan_config.epochs):
                for i, (imgs, _) in enumerate(dataloader):
                    
                    # Ground truths
                    valid = torch.ones(imgs.size(0), 1, device=device)
                    fake = torch.zeros(imgs.size(0), 1, device=device)

                    real_imgs = imgs.to(device)

                    # -----------------
                    #  Train Generator
                    # -----------------
                    optimizer_G.zero_grad()
                    z = torch.randn(imgs.size(0), self.gan_config.latent_dim, device=device)
                    gen_imgs = generator(z)
                    g_loss = adversarial_loss(discriminator(gen_imgs), valid)
                    g_loss.backward()
                    optimizer_G.step()

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    optimizer_D.zero_grad()
                    real_loss = adversarial_loss(discriminator(real_imgs), valid)
                    fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                    d_loss = (real_loss + fake_loss) / 2
                    d_loss.backward()
                    optimizer_D.step()

                logging.info(f"[Epoch {epoch}/{self.gan_config.epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

            # Save the Generator (the component used for generating new fashion)
            torch.save(generator.state_dict(), self.gan_config.trained_model_path_gen)
            logging.info("GAN Generator model saved successfully")

            return self.gan_config.trained_model_path_gen

        except Exception as e:
            raise CustomException(e, sys)