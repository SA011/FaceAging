import os
import random
from argparse import ArgumentParser
import yaml
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from Models import MultiResUNet

parser = ArgumentParser()
parser.add_argument('--image_dir', default='./archive/test_image/', help='The image directory')
parser.add_argument('--checkpoint_dir', default='./pretrained/', help='The image directory')
parser.add_argument('--config', default='configs/aging_gan.yaml', help='Config to use for training')

@torch.no_grad()
def main():
    args = parser.parse_args()
    with open(args.config) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    image_dir_O = args.image_dir + 'testO/'
    image_dir_Y = args.image_dir + 'testY/'
    old_image_paths = [os.path.join(image_dir_O, x) for x in os.listdir(image_dir_O) if
                   x.endswith('.png') or x.endswith('.jpg')]
    young_image_paths = [os.path.join(image_dir_Y, x) for x in os.listdir(image_dir_Y) if
                   x.endswith('.png') or x.endswith('.jpg')]
    
    model = MultiResUNet(3, 3, configs['gen_alpha'], configs['ngf'])
    ckpt = torch.load(args.checkpoint_dir + configs['y2o'], map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    nr_images = len(young_image_paths) 
    fig, ax = plt.subplots(2, nr_images, figsize=(20, 10))
    random.shuffle(young_image_paths)
    for i in range(nr_images):
        img = Image.open(young_image_paths[i]).convert('RGB')
        img = trans(img).unsqueeze(0)
        aged_face = model(img)
        aged_face = (aged_face.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
        ax[0, i].imshow((img.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0)
        ax[1, i].imshow(aged_face)
    # plt.show()
    plt.savefig("mygraph_y2o.png")





    model = MultiResUNet(3, 3, configs['gen_alpha'], configs['ngf'])
    ckpt = torch.load(args.checkpoint_dir + configs['o2y'], map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    nr_images = len(old_image_paths) 
    fig, ax = plt.subplots(2, nr_images, figsize=(20, 10))
    random.shuffle(old_image_paths)
    for i in range(nr_images):
        img = Image.open(old_image_paths[i]).convert('RGB')
        img = trans(img).unsqueeze(0)
        aged_face = model(img)
        aged_face = (aged_face.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
        ax[0, i].imshow((img.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0)
        ax[1, i].imshow(aged_face)
    # plt.show()
    plt.savefig("mygraph_o2y.png")

if __name__ == '__main__':
    main()
