import torch
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader

np.random.seed(1)
torch.manual_seed(1)

class histoCancerDataset(Dataset):
    def __init__(self, data_dir, transform,data_type="train"):      
        path2data=os.path.join(data_dir, data_type)
        self.filenames = os.listdir(path2data)
        self.full_filenames = [os.path.join(path2data, f) for f in self.filenames]
        csv_filename=data_type+"_labels.csv"
        path2csvLabels=os.path.join(data_dir,csv_filename)
        labels_df=pd.read_csv(path2csvLabels)
        labels_df.set_index("id", inplace=True)
        self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in self.filenames]
        self.transform = transform
      
    def __len__(self):
        return len(self.full_filenames)
      
    def __getitem__(self, idx):
        image = Image.open(self.full_filenames[idx])
        image = self.transform(image)
        return image, self.labels[idx]

data_dir = "../chapter2/data/"
data_transformer = transforms.Compose([transforms.ToTensor()])   
hist_ds = histoCancerDataset(data_dir, data_transformer,data_type="train")

test_index = np.random.randint(hist_ds.__len__(),size= 100)
test_ds = Subset(hist_ds,test_index)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)  
    