import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

IMG_EXTENSIONS = ["png", "jpg"]

class ImagetoImageDataset(Dataset):
    def __init__(self, domain_dirs, transforms=None):
        self.images = []
        for domain_dir in domain_dirs:
            images = [os.path.join(domain_dir, x) for x in os.listdir(domain_dir) if x.lower().endswith(tuple(IMG_EXTENSIONS))]
            self.images.extend(images)

        self.transforms = transforms
        self.length = len(self.images)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx % self.length
        image_path = self.images[idx]
        image = np.array(Image.open(image_path).convert("RGB"))

        if self.transforms is not None:
            image = self.transforms(image)

        return image, self.get_age_label(image_path)

    def get_age_label(self, image_path):
        # Implement a function to extract age label from image path
        # This function will depend on how your image directories are structured
        # For example, you might extract age label from the directory name
        # Modify this according to your specific dataset structure
        # For simplicity, let's assume age is encoded in the directory name
        age_label = os.path.basename(os.path.dirname(image_path))
        age_label = int(age_label.split('-')[0])
        return age_label
