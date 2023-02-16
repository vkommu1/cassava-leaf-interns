import torch
import pandas as pd
from PIL import Image as img
import torchvision.transforms as t


class StartingDataset(torch.utils.data.Dataset):
    """
    Load 800 x 600 images
    """

    def __init__(self, path):

        df = pd.read_csv(path)
        self.image_id = df['image_id']
        self.labels = df['label']

    def __getitem__(self, index):

        id = self.image_id.iloc[index]
        label = torch.tensor(int(self.labels.iloc[index]))

        img_path = '../dataset/train_images/' + id

        image = img.open(img_path).resize((800, 600))
        
        return (t.ToTensor()(image), label)

    def __len__(self):

        return len(self.labels)

