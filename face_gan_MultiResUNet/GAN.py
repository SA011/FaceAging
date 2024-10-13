import pytorch_lightning as pl
import torch
from Models import MultiResUNet, Discriminator
import torch.nn.functional as F
from torchvision.utils import make_grid
import itertools
from torchvision import transforms
from Dataset import ImagetoImageDataset
from torch.utils.data import DataLoader

class GAN(pl.LightningModule):
    def load_from_checkpoint(self, checkpoint_path):
        y2o = checkpoint_path + self.hparams['y2o']
        o2y = checkpoint_path + self.hparams['o2y']
        self.genY2O.load_state_dict(torch.load(y2o))
        self.genO2Y.load_state_dict(torch.load(o2y))

    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(hparams)
        self.genY2O = MultiResUNet(3, 3, self.hparams['gen_alpha'], self.hparams['ngf'])
        self.genO2Y = MultiResUNet(3, 3, self.hparams['gen_alpha'], self.hparams['ngf'])
        self.disY = Discriminator(3, self.hparams['dis_alpha'], self.hparams['ndf'])
        self.disO = Discriminator(3, self.hparams['dis_alpha'], self.hparams['ndf'])

        # cache for generated images
        self.generated_Y = None
        self.generated_O = None
        self.real_Y = None
        self.real_O = None

    def forward(self, x):
        return self.genY2O(x)
    
    def training_step(self, batch, batch_idx):
        g_optim, d_optim = self.optimizers()
        g_optim.zero_grad()
        
        real_Y, real_O = batch


        fake_O = self.genY2O(real_Y)
        pred_O = self.disO(fake_O)
        loss_Y2O = F.binary_cross_entropy(pred_O, torch.ones(pred_O.shape).type_as(pred_O))

        rec_Y = self.genO2Y(fake_O)
        loss_Y2O2Y = F.mse_loss(rec_Y, real_Y)

        real_GY = self.genO2Y(real_Y)
        loss_Y2Y = F.mse_loss(real_GY, real_Y)



        fake_Y = self.genO2Y(real_Y)
        pred_Y = self.disY(fake_Y)
        loss_O2Y = F.binary_cross_entropy(pred_Y, torch.ones(pred_Y.shape).type_as(pred_Y))

        rec_O = self.genY2O(fake_Y)
        loss_O2Y2O = F.mse_loss(rec_O, real_O)

        real_GO = self.genY2O(real_O)
        loss_O2O = F.mse_loss(real_GO, real_O)
        

        g_loss = (loss_Y2O + loss_O2Y) * self.hparams['adv_weight'] + (loss_Y2Y + loss_O2O) * self.hparams['identity_weight'] + (loss_Y2O2Y + loss_O2Y2O) * self.hparams['cycle_weight']


        outputG = {
            'loss': g_loss,
            'log': {'Loss/Generator': g_loss.detach()}
        }

        self.log('Loss/Generator', g_loss.detach())
        

        # Log to tb
        if batch_idx % 500 == 0:
            self.genY2O.eval()
            self.genY2O.eval()
            fake_Y = self.genO2Y(real_O)
            fake_O = self.genY2O(real_Y)
            self.logger.experiment.add_image('Real/Y', make_grid(real_Y, normalize=True, scale_each=True),
                                                self.current_epoch)
            self.logger.experiment.add_image('Real/O', make_grid(real_O, normalize=True, scale_each=True),
                                                self.current_epoch)
            self.logger.experiment.add_image('Generated/Y',
                                                make_grid(fake_Y, normalize=True, scale_each=True),
                                                self.current_epoch)
            self.logger.experiment.add_image('Generated/O',
                                                make_grid(fake_O, normalize=True, scale_each=True),
                                                self.current_epoch)
            self.genY2O.train()
            self.genO2Y.train()

            output_path = './pretrained/'
            torch.save(self.genY2O.state_dict(), f"{output_path}{self.hparams['y2o']}")
            torch.save(self.genO2Y.state_dict(), f"{output_path}{self.hparams['o2y']}")

        self.manual_backward(g_loss)
        g_optim.step()

        d_optim.zero_grad()
        pred_RY = self.disY(real_Y)
        loss_RY = F.binary_cross_entropy(pred_RY, torch.ones(pred_RY.shape).type_as(pred_RY))

        pred_RO = self.disY(real_O)
        loss_RO = F.binary_cross_entropy(pred_RO, torch.ones(pred_RO.shape).type_as(pred_RO))

        pred_FY = self.disY(self.genO2Y(real_O).detach())
        loss_FY = F.binary_cross_entropy(pred_FY, torch.ones(pred_FY.shape).type_as(pred_FY))

        pred_FO = self.disO(self.genY2O(real_Y).detach())
        loss_FO = F.binary_cross_entropy(pred_FO, torch.ones(pred_FO.shape).type_as(pred_FO))

        d_loss = loss_RO + loss_FO + loss_RY + loss_FY

        outputD = {
            'loss': d_loss,
            'log': {'Loss/Discriminator': d_loss.detach()}
        }
        self.log('Loss/Discriminator', d_loss.detach())

        self.manual_backward(d_loss)
        d_optim.step()

        

    def configure_optimizers(self):
        g_optim = torch.optim.Adam(itertools.chain(self.genY2O.parameters(), self.genO2Y.parameters()),
                                   lr=self.hparams['lr'], betas=(0.5, 0.999),
                                   weight_decay=self.hparams['weight_decay'])
        d_optim = torch.optim.Adam(itertools.chain(self.disY.parameters(),
                                                   self.disO.parameters()),
                                   lr=self.hparams['lr'],
                                   betas=(0.5, 0.999),
                                   weight_decay=self.hparams['weight_decay'])
        return [g_optim, d_optim], []
    

    def train_dataloader(self):
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.hparams['img_size'] + 50, self.hparams['img_size'] + 50)),
            transforms.RandomCrop(self.hparams['img_size']),
            #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            #transforms.RandomPerspective(p=0.5),
            transforms.RandomRotation(degrees=(0, int(self.hparams['augment_rotation']))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        dataset = ImagetoImageDataset(self.hparams['domainY_dir'], self.hparams['domainO_dir'], train_transform)
        #use small data
        print(f"Using {len(dataset)} images for training")
        # dataset = torch.utils.data.Subset(dataset, range(0, 10))

        return DataLoader(dataset,
                          batch_size=self.hparams['batch_size'],
                          num_workers=self.hparams['num_workers'],
                          shuffle=True)

