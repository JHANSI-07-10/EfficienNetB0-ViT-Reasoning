import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class HAM10000(Dataset):
    def __init__(self, df, img_dir, transform=None):
        """
        Initializes the dataset with metadata and image directory.
        """
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
        
        self.label_map = {
            'akiec': 0,
            'bcc': 1,   
            'bkl': 2,   
            'df': 3,   
            'mel': 4,   
            'nv': 5,   
            'vasc': 6   
        }

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Loads and returns a single sample from the dataset at the given index.
        """
       
        img_id = self.df.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        
       
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find image: {img_path}")
        
      
        label_name = self.df.iloc[idx]['dx']
        label = self.label_map[label_name]
        
       
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label)
    


if __name__ == "__main__":
    from torchvision import transforms
    
    print("--- Testing dataset.py ---")
    
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata_path = os.path.join(BASE_DIR, "Data", "HAM10000_metadata.csv")
    image_dir = os.path.join(BASE_DIR, "Data", "all_images")

    print(f"Checking for CSV at: {metadata_path}")
    
    if not os.path.exists(metadata_path):
        print(f" Error: Metadata file not found at {metadata_path}")
    else:
        try:
            
            df = pd.read_csv(metadata_path)
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        
            ds = HAM10000(df, image_dir, transform=test_transform)
            
           
            img, lbl = ds[0]
            
            print(f"✅ Success!")
            print(f"Number of samples in test: {len(ds)}")
            print(f"Image tensor shape: {img.shape}")
            print(f"Label: {lbl.item()}")
            print(df.head())
            
        except Exception as e:
            print(f" An error occurred: {e}")