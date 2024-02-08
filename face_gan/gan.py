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
import numpy as np
import scipy.io
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--load', default=0, type=int, help='load model', required=False)
argparser.add_argument('--save', default=1, type=int, help='save model', required=False) 
argparser.add_argument('--train', default=1, type=int, help='train model', required=False)

args = argparser.parse_args()

random_seed = 42
torch.manual_seed(random_seed)

device = torch.device('cuda' if not torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 32
num_epochs = 20
latent_size = 128
g_learning_rate = 0.005
d_learning_rate = 0.002
limit = batch_size * 200
image_size = 64

# Load Data
dir = 'wiki_crop/'
mat = scipy.io.loadmat(dir + 'wiki.mat')
wiki = mat['wiki']
wiki = wiki[0][0]

dob = wiki[0][0]
photo_taken = wiki[1][0]
full_path = wiki[2][0]
gender = wiki[3][0]
name = wiki[4][0]
face_location = wiki[5][0]
face_score = wiki[6][0]
second_face_score = wiki[7][0]

print(photo_taken.shape)
age = []
for i in range(len(dob)):
    birth_year = int(str(dob[i])[0:2])
    if birth_year > 20:
        birth_year += 1900
    else:
        birth_year += 2000
    age.append(photo_taken[i] - birth_year)
    if age[i] < 0:
        age[i] += 100
age = torch.tensor(age)

# sort by face_score
face_score = torch.tensor(face_score)
face_score, indices = torch.sort(face_score, descending=True)
age = age[indices]
full_path = full_path[indices]
face_location = face_location[indices]
face_location_int = []
for i in range(len(face_location)):
    face_location_int.append([])
    for j in range(len(face_location[i][0])):
        face_location_int[i].append(int(face_location[i][0][j]))

# print(type(face_location_int[0][0]))
face_location = face_location_int

# Load Images
all_images = []
error_count = 0
size = 0
for i in range(min(len(full_path), limit)):
    try:
        image = plt.imread(dir + full_path[i][0])
        # print(image.shape)
        all_images.append(image)
        # size += image.shape[0] * image.shape[1] * image.shape[2]
    except:
        error_count += 1
        all_images.append(all_images[i - 1])

print(error_count)
print(len(all_images))
# print(size)
# exit(0)
# print(face_location.shape)
# Preprocess Images
def test(image):
    image = torch.tensor(image).cpu()
    image = image.reshape(3, image_size, image_size)
    image = image.permute(1, 2, 0)
    image = image * 255
    image = image.numpy()
    image = image.astype(np.uint8)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


for i in range(len(all_images)):
    temp = all_images[i][face_location[i][1]:face_location[i][3], face_location[i][0]:face_location[i][2]]
    if temp.shape[0] == 0 or temp.shape[1] == 0:
        temp = all_images[i]
    all_images[i] = temp

def preprocess(image):
    
    while image.shape[0] == 0 or image.shape[1] == 0:
        image = all_images[np.random.randint(0, len(all_images) - 1)]
    
    
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    if image.shape[2] >= 4:
        image = image[:, :, :3]
    # print(image.shape)
    # resize image to image_size x image_size
    image = torch.tensor(image)
    image = image.permute(2, 0, 1)
    image = image.float()
    image = image / 255
    # print(image.shape)
    image = F.interpolate(image.unsqueeze(0), size=image_size, mode='bilinear', align_corners=False).squeeze(0)
    # print(image.shape)
    # test(image)
    return image

all_images = [preprocess(image) for i, image in enumerate(all_images)]
# exit(0)
# Create Dataset
class WikiDataset(torch.utils.data.Dataset):
    def __init__(self, images, age):
        self.images = images
        self.age = age

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.age[index]

dataset = WikiDataset(all_images, age)
train_dataset = dataset
# Create DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # input: 3x64x64
        self.layers = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=4, stride=2), # 10x31x31
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=4, stride=2), # 20x14x14
            nn.ReLU(),
            nn.Conv2d(20, 25, kernel_size=4, stride=2), # 25x6x6
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(25 * 6 * 6, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)
    
        


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            # nn.Linear(latent_size, 64 * 4 * 4),
            # nn.ReLU(),
            # nn.Unflatten(1, (64, 4, 4)),
            # nn.ConvTranspose2d(64, 32, kernel_size=6, stride=3),
            # nn.ReLU(),
            # nn.ConvTranspose2d(32, 16, kernel_size=7, stride=4),
            # nn.ReLU(),
            # nn.ConvTranspose2d(16, 3, kernel_size=8, stride=4)
            #output: 3x64x64, input: 128, use less layers
            nn.Linear(latent_size, 64 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # 32x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),    # 16x32x32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),     # 3x64x64
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


discriminator = Discriminator().to(device)
generator = Generator().to(device)
best_loss_D = -1
best_loss_G = -1

# load model
if args.load:
    discriminator.load_state_dict(torch.load('discriminator.ckpt'))
    generator.load_state_dict(torch.load('generator.ckpt'))

# Loss and optimizer
criterion = nn.BCELoss()
d_optimizer = optim.Adam(discriminator.parameters(), lr=d_learning_rate)
g_optimizer = optim.Adam(generator.parameters(), lr=g_learning_rate)

# schdular
d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=5, gamma=0.9)
g_scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=5, gamma=0.9)


# Train the model
total_step = len(train_loader)
d_losses, g_losses = [], []


def plot():
    with torch.no_grad():
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = generator(z).cpu()
        fake_images = fake_images.reshape(fake_images.size(0), 3, image_size, image_size)
        fake_images = fake_images.permute(0, 2, 3, 1)
        fake_images = fake_images * 255
        fake_images = fake_images.numpy()
        fake_images = fake_images.astype(np.uint8)
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(fake_images[i])
            plt.axis('off')
        plt.show()

if args.train:
    for epoch in range(num_epochs):
        if (epoch) % 5 == 0:
            plot()
            if args.save:
                torch.save(generator.state_dict(), 'generator.ckpt')
                torch.save(discriminator.state_dict(), 'discriminator.ckpt')
                
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
            if best_loss_D == -1:
                best_loss_D = d_loss.item()
                best_loss_G = g_loss.item()

            if (i + 1) % 100 == 0:
                print("epoch {}/{}, step {}/{}, d_loss = {:.4f}, g_loss = {:.4f}".format(epoch + 1, num_epochs, i + 1, total_step, d_loss.item(), g_loss.item()))
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())


        d_scheduler.step()
        g_scheduler.step()

# Test the model
plot()


# Save the model checkpoint
# if args.save:
#     torch.save(generator.state_dict(), 'generator.ckpt')
#     torch.save(discriminator.state_dict(), 'discriminator.ckpt')

plt.plot(d_losses, label='Discriminator')
plt.plot(g_losses, label='Generator')
plt.legend()
plt.show()


