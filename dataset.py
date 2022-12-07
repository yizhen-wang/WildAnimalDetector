import os
import PIL as pil
import util as util
from PIL import Image
from torch.utils.data import Dataset

class Spotted_Skunk_Test(Dataset):
    def __init__(self, data_dir, data_list, cate_list, transform=None):
        self.transform  = transform
        self.data_dir   = data_dir
        self.image_list = data_list
        self.cate_list  = cate_list
        
    def __len__(self):
        return (len(self.image_list))
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_list[idx])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        
        return image




