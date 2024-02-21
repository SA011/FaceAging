import os
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from gan_module import Generator

parser = ArgumentParser()
parser.add_argument(
    '--image_dir', default='../face_gan/test/', help='The image directory')


@torch.no_grad()
def main():
    args = parser.parse_args()
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir) if
                   x.endswith('.png') or x.endswith('.jpg')]
    model = Generator(ngf=32, n_residual_blocks=9)
    ckpt = torch.load('./model_A2B.pth', map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    nr_images = len(image_paths) if len(image_paths) >= 6 else 6
    fig, ax = plt.subplots(2, nr_images, figsize=(20, 10))
    random.shuffle(image_paths)
    for i in range(nr_images):
        img = Image.open(image_paths[i]).convert('RGB')
        img = trans(img).unsqueeze(0)
        age_labels = torch.tensor([random.randint(15, 70)]).unsqueeze(0)
        expanded_age_labels = age_labels.view(len(img), -1, 1, 1).float().repeat(1, 1, 256, 256)

        aged_face = model(torch.cat((img, expanded_age_labels), dim=1))         
        aged_face = (aged_face.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
        ax[0, i].imshow((img.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0)
        ax[1, i].imshow(aged_face)
    # plt.show()
    plt.savefig("mygraph.png")


if __name__ == '__main__':
    main()
