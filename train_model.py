"""Train a Variational Autoencoder (VAE) on MNIST and save the weights."""
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

LATENT_DIM = 16
EPOCHS = 20
BATCH_SIZE = 128
LATENT_CHOICES = [2, 4, 8, 16]
MODEL_PATH = "vae_model.pth"


class VAE(nn.Module):
    """Encoder 784 -> 256 -> 128 -> (mu, logvar) of 16 dims; decoder mirrors it."""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 784)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    @staticmethod
    def mask(z, keep):
        """Keep the first `keep` latent values, set the rest to zero."""
        if keep is None or keep >= z.size(1):
            return z
        z = z.clone()
        z[:, keep:] = 0.0
        return z

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x, keep=None):
        mu, logvar = self.encode(x)
        z = self.mask(self.reparameterize(mu, logvar), keep)
        return self.decode(z), mu, logvar


def vae_loss(recon, target, mu, logvar):
    bce = F.binary_cross_entropy(recon, target, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kld) / target.size(0)


def add_noise(x, level):
    return torch.clamp(x + torch.randn_like(x) * level, 0.0, 1.0)


def main():
    torch.manual_seed(0)
    random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST("data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1000)

    model = VAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training: noisy input -> clean target (denoising), and random latent
    # truncation so the decoder works with 2, 4, 8 or 16 retained dimensions.
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for x, _ in train_loader:
            x = x.view(-1, 784).to(device)
            noisy = add_noise(x, random.uniform(0.0, 0.5))
            recon, mu, logvar = model(noisy, keep=random.choice(LATENT_CHOICES))
            loss = vae_loss(recon, x, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        print(f"Epoch {epoch:02d}/{EPOCHS}  loss: {total / len(train_ds):.2f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved weights to {MODEL_PATH}")

    # Evaluation on the test set (noise level 0.3)
    model.eval()
    x_test, _ = next(iter(test_loader))
    x_test = x_test.view(-1, 784).to(device)
    torch.manual_seed(1)
    noisy_test = add_noise(x_test, 0.3)
    print("\nTest-set evaluation (noise = 0.30)")
    print(f"Noisy-input MSE: {F.mse_loss(noisy_test, x_test).item():.4f}")
    with torch.no_grad():
        for k in LATENT_CHOICES:
            mu, _ = model.encode(noisy_test)
            recon = model.decode(model.mask(mu, k))
            print(f"Latent dims {k:2d}: reconstruction MSE = "
                  f"{F.mse_loss(recon, x_test).item():.4f}")

    # Save comparison image
    with torch.no_grad():
        mu, _ = model.encode(noisy_test[:8])
        recon = model.decode(mu).cpu()
    rows = [("Original", x_test[:8].cpu()), ("Noisy (0.30)", noisy_test[:8].cpu()),
            ("Reconstructed", recon)]
    fig, axes = plt.subplots(3, 8, figsize=(12, 5))
    for r, (name, imgs) in enumerate(rows):
        for c in range(8):
            axes[r, c].imshow(imgs[c].view(28, 28), cmap="gray")
            axes[r, c].axis("off")
        axes[r, 0].set_title(name, loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig("training_results.png", dpi=120)
    print("Saved training_results.png")

    # Save one sample test digit per class for the app
    os.makedirs("samples", exist_ok=True)
    seen = set()
    for img, label in test_ds:
        if label not in seen:
            seen.add(label)
            transforms.ToPILImage()(img).save(f"samples/digit_{label}.png")
        if len(seen) == 10:
            break
    print("Saved sample digits to samples/")


if __name__ == "__main__":
    main()
