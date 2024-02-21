import itertools
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from dataset import ImagetoImageDataset
from models import Generator, Discriminator

class AgingGAN(pl.LightningModule):

    def __init__(self, hparams):
        super(AgingGAN, self).__init__()
        self.save_hyperparameters(hparams)
        self.generator = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
        self.discriminator = Discriminator(hparams['ndf'])

        # cache for generated images
        self.generated_images = None
        self.real_images = None

    @property
    def automatic_optimization(self):
        return False

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        real_images, age_labels = batch
        # print(age_labels.shape)
        expanded_age_labels = age_labels.view(len(real_images), -1, 1, 1).float().repeat(1, 1, 256, 256)
        # print(expanded_age_labels)
        fake_images = self.generator(torch.cat((real_images, expanded_age_labels), dim=1))

        # Generator loss
        pred_fake = self.discriminator(torch.cat((fake_images, expanded_age_labels), dim=1))
        loss_G = F.mse_loss(pred_fake, torch.ones_like(pred_fake)) * self.hparams['adv_weight']
        loss_identity = F.l1_loss(fake_images, real_images) * self.hparams['identity_weight']

        # Discriminator loss
        pred_real = self.discriminator(torch.cat((real_images, expanded_age_labels), dim=1))
        loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real))
        pred_fake = self.discriminator(torch.cat((fake_images.detach(), expanded_age_labels), dim=1))
        loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        # Total loss
        loss_total = loss_G + loss_identity + loss_D

        output = {
            'loss': loss_total,
            'log': {'Loss/Generator': loss_G, 'Loss/Identity': loss_identity, 'Loss/Discriminator': loss_D}
        }
        self.log('Loss/Generator', loss_G)
        self.log('Loss/Identity', loss_identity)
        self.log('Loss/Discriminator', loss_D)

        # Update cache
        self.generated_images = fake_images
        self.real_images = real_images

        # Log to Tensorboard
        if batch_idx % 500 == 0:
            self.logger.experiment.add_image('Real', make_grid(real_images, normalize=True, scale_each=True), self.current_epoch)
            self.logger.experiment.add_image('Generated', make_grid(fake_images, normalize=True, scale_each=True), self.current_epoch)

        return output

    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.hparams['lr'], betas=(0.5, 0.999), weight_decay=self.hparams['weight_decay'])
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams['lr'], betas=(0.5, 0.999), weight_decay=self.hparams['weight_decay'])
        return [optimizer_G, optimizer_D], []

    def train_dataloader(self):
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.hparams['img_size'] + 50, self.hparams['img_size'] + 50)),
            transforms.RandomCrop(self.hparams['img_size']),
            transforms.RandomRotation(degrees=(0, int(self.hparams['augment_rotation']))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        dataset = ImagetoImageDataset(self.hparams['domain_dirs'], train_transform)

        return DataLoader(dataset, batch_size=self.hparams['batch_size'], num_workers=self.hparams['num_workers'], shuffle=True)

# # Example usage:
# age_gan = AgingGAN(hparams)
# trainer = pl.Trainer()
# trainer.fit(age_gan)
