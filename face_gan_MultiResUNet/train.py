import argparse
import torch
import yaml
import os
from GAN import GAN
from pytorch_lightning import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/aging_gan.yaml', help='Config to use for training')
parser.add_argument('--load_checkpoint_dir', default=None, help='Path to load checkpoint')
parser.add_argument('--save_checkpoint_dir', default=None, help='Path to save checkpoint')

def main():
    args = parser.parse_args()
    with open(args.config) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    configs['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('medium')
    print(configs)
    model = GAN(configs)
    if args.load_checkpoint_dir:
        model.load_checkpoint(args.load_checkpoint_dir)

    trainer = Trainer(max_epochs=configs['epochs'])
    trainer.fit(model)

    output_path = args.save_checkpoint_dir if args.save_checkpoint_dir else 'pretrained/'
    try:
        os.mkdir(output_path)
    except:
        pass
    torch.save(model.genY2O.state_dict(), f"{output_path}{configs['y2o']}")
    torch.save(model.genO2Y.state_dict(), f"{output_path}{configs['o2y']}")



main()