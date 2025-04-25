import os
import random
from PIL import Image
import ast
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import ast

# Define transforms for training validation and test datasets
# Maybe we can use a smaller resize 
default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class BrailleDataset(Dataset):
    def __init__(self, transform=default_transform, mode = "train", file_path=None):
        self.transform = transform
        self.samples = pd.read_csv(file_path)
        self.mode = mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name,label = self.samples.iloc[idx]
        label = ast.literal_eval(label)
        image = cv2.imread(img_name)  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = Image.fromarray(image)  # Convert to PIL for transforms
        if self.transform:
            image = self.transform(image)

        # only when the mode is train, we can apply the flip
        if self.mode == "train":
            # get random probability
            p_hflip = random.random()
            p_vflip = random.random()
            p_rotate = random.random()
            # vertical flip
            if p_vflip > 0.5:
                image = torch.flip(image, [1])
                # half the label
                half = label[: len(label) // 2]
                another_half = label[len(label) // 2 :]
                label = half[::-1] + another_half[::-1]

            # horizontal flip
            if p_hflip > 0.5:
                image = torch.flip(image, [2])
                # reverse the label
                half = label[: len(label) // 2]
                another_half = label[len(label) // 2 :]
                label = another_half + half
            
            # random rotation
            if p_rotate > 0.5:
                angle = random.randint(-10, 10)
                image = transforms.functional.rotate(image, angle)
        
        return image, torch.tensor(label,dtype=torch.float)