import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

random_seed = 42
torch.manual_seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 64
num_epochs = 20
latent_size = 128
learning_rate = 0.0001

# Load Data

train_dataset = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_dataset, val_dataset = random_split(train_dataset, [55000, 5000])
test_dataset = MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(20 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_size, 128 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=7)
        )

    def forward(self, x):
        return self.layers(x)


discriminator = Discriminator().to(device)
generator = Generator().to(device)

# load model
discriminator.load_state_dict(torch.load('discriminator.ckpt'))
generator.load_state_dict(torch.load('generator.ckpt'))

# Loss and optimizer
criterion = nn.BCELoss()
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
d_losses, g_losses = [], []

def plot():
    with torch.no_grad():
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = generator(z).cpu()
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.imshow(fake_images[i][0], cmap='gray')
        plt.show()


for epoch in range(num_epochs):
    if (epoch) % 5 == 0:
        plot()
    for i, (images, _) in enumerate(train_loader):
        
        # train generator
        g_optimizer.zero_grad()
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, torch.ones_like(outputs))
        g_loss.backward()
        g_optimizer.step()

        # train discriminator
        d_optimizer.zero_grad()
        real_images = images.to(device)
        outputs = discriminator(real_images)
        real_loss = criterion(outputs, torch.ones_like(outputs))
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = generator(z).detach()
        outputs = discriminator(fake_images)
        fake_loss = criterion(outputs, torch.zeros_like(outputs))
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        d_optimizer.step()

        if (i + 1) % 800 == 0:
            print("epoch {}/{}, step {}/{}, d_loss = {:.4f}, g_loss = {:.4f}".format(epoch + 1, num_epochs, i + 1, total_step, d_loss.item(), g_loss.item()))
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

# Test the model
plot()

# Save the model checkpoint
torch.save(generator.state_dict(), 'generator.ckpt')
torch.save(discriminator.state_dict(), 'discriminator.ckpt')

plt.plot(d_losses, label='Discriminator')
plt.plot(g_losses, label='Generator')
plt.legend()
plt.show()


