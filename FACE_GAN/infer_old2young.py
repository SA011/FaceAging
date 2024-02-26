import os
import random
import yaml
from argparse import ArgumentParser
from deepface import DeepFace
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from gan_module import Generator
import sys

parser = ArgumentParser()

parser.add_argument(
    '--image_dir', default='./archive/preprocessed/trainB', help='The image directory')
parser.add_argument(
    '--checkpoint_dir', default='./pretrained_models/OldToYoung.pth', help='The image directory')

parser.add_argument(
    '--config', default='./configs/aging_gan.yaml', help='The Config File')

@torch.no_grad()
def main():
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir) if
                   x.endswith('.png') or x.endswith('.jpg')]
    model = Generator(ngf=config['ngf'], n_residual_blocks=9)
    ckpt = torch.load(args.checkpoint_dir, map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    print(len(image_paths))
    nr_images = min(10, len(image_paths))
    count = 0
    distance = 0
    age_gap = 0
    for i in range(nr_images):
        img = Image.open(image_paths[i]).convert('RGB')
        img = trans(img).unsqueeze(0)
        aged_face = model(img)
        aged_face = (aged_face.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
        # save the aged face as a png file
        aged_face = Image.fromarray((aged_face * 255).astype('uint8'))
        aged_face.save(f'./archive/aged_faces/aged_face_{i}.png')
        result = DeepFace.verify(image_paths[i], f'./archive/aged_faces/aged_face_{i}.png', enforce_detection=False)
        distance += result['distance']
        count += result['verified'] == True
        print(result)
        a1 = DeepFace.analyze(image_paths[i], actions=['age'])
        a2 = DeepFace.analyze(f'./archive/aged_faces/aged_face_{i}.png', actions=['age'])
        age_gap += a1['age'] - a2['age']

    print(f'Average distance: {distance/nr_images}')
    print(f"Verified: {count}/{nr_images} images")
    print(f"Accuracy: {count/nr_images*100}%")
    print(f'Average age gap: {age_gap/nr_images}')
    
    # fig, ax = plt.subplots(2, nr_images, figsize=(20, 10))
    # random.shuffle(image_paths)
    # for i in range(nr_images):
    #     img = Image.open(image_paths[i]).convert('RGB')
    #     img = trans(img).unsqueeze(0)
    #     aged_face = model(img)
    #     aged_face = (aged_face.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
    #     ax[0, i].imshow((img.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0)
    #     ax[1, i].imshow(aged_face)
    # # plt.show()
    # plt.savefig("mygraph.png")


if __name__ == '__main__':
    main()
