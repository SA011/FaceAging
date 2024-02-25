from argparse import ArgumentParser

import yaml
from pytorch_lightning import Trainer
import torch

from gan_module import AgingGAN

parser = ArgumentParser()
parser.add_argument('--config', default='configs/aging_gan.yaml', help='Config to use for training')
parser.add_argument('--load_checkpoint_dir', default=None, help='Path to load checkpoint')
parser.add_argument('--save_checkpoint_dir', default=None, help='Path to save checkpoint')



def main():
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)
    model = AgingGAN(config)
    if args.load_checkpoint_dir:
        model.load_from_checkpoint(args.load_checkpoint_dir)
        
    trainer = Trainer(max_epochs=config['epochs'], gpus=config['gpus'], auto_scale_batch_size='binsearch')
    trainer.fit(model)
    #save models
    output_path = args.save_checkpoint_dir if args.save_checkpoint_dir else 'pretrained_model/'
    torch.save(model.genA2B.state_dict(), f"{output_path}YoungToOld.pth")
    torch.save(model.genB2A.state_dict(), f"{output_path}OldToYoung.pth")

if __name__ == '__main__':
    main()
