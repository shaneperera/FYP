import os
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader

# Study level data pipeline

# Define the data categories (Training or testing)
data_cat = ['train', 'valid']


# Create function to create dictionary regarding pathway to specific study type, number of patients in study type and
def get_study_data(study_type):
    """
    Returns a dictionary, with keys 'train' and 'valid' and respective values as study level data frames, these data frames
    contain three columns:
         Path: Path directory to data
         Count: Total number of studies for that patient (eg. can have multiple positive or negative case)
         Label: 0 or 1 to represent positive classification or not
    Inputs:
          Study_type (string): one of the seven study type folder names in 'train/valid' dataset
    """

    # Define empty dictionary
    study_data = {}

    # Define study classification dictionary --> Reference key to determine classification
    study_label = {'positive': 1, 'negative': 0}

    # Populate dictionary based on whether it is the valid/train data set
    # Example of directory to MURA data set -->  C:\Users\shann\Documents\FYP\MURA-v1.1\train\...
    for category in data_cat:

        # Locate the MURA dataset
        # Include directory extension --> Special type of string formatting
        base_directory = 'C:/Users/shann/Documents/FYP/MURA-v1.1/%s/%s' % (category, study_type)

        # List of all the patients inside the study level
        # os.walk --> Generates file names in the directory --> When you enter XR_ELBOW --> Will print directory first
        # [0][1] Indexes the first element, but second element of the first element
        patients = list(os.walk(base_directory))[0][1]

        # Each category (train/valid) will have 3 columns (the key is based on the path directory)
        study_data[category] = pd.DataFrame(columns=['Path', 'Count', 'Label'])

        # Populate the rows (path,count,label) of the pandas dataframe
        i = 0  # Used to index the rows of the pandas data frame (table)
        for patient in tqdm(patients):  # Create a loading bar when populating the table
            # os.listdir --> Returns a list of files in the directory
            # (difference with os.walk is that you can't specify the direction)
            for study in os.listdir(base_directory + patient):
                # Chance of study1_negative & study2_positive (for eg.)
                path = base_directory + '/' + patient + '/' + study
                label = study_label[study.split('_')[1]]  # Notation: study1_negative -> Splits --> Choose +ve or -ve
                study_data[category].loc[i] = [path, len(os.listdir(path)), label]  # add new row to data frame
                i += 1
    return study_data


class ImageDataset(Dataset):
    """Training dataset."""

    def __init__(self, df, transform=None):
        """
        Args:
            df = Pandas data frame with image path and labels (pd.DataFrame)
            transform (callable, optional): Optional transform to be applied on a sample x-ray.
        """

        self.df = df
        self.transform = transform  # Transforms include rotations / scaling / etc

    def __len__(self):
        # Returns the length of dataframe
        return len(self.df)

    def __getitem__(self, idx):
        # Returns an element based on the index (idx)
        study_path = self.df.iloc[idx, 0]
        count = self.df.iloc[idx, 1]
        images = []
        for i in range(count):
            image = pil_loader(study_path + 'image%s.png' % (i + 1))
            images.append(self.transform(image))
        images = torch.stack(images)
        label = self.df.iloc[idx, 2]
        sample = {'images': images, 'label': label}
        return sample

    def get_dataloaders(self, batch_size=8, study_level=False):
        """
        Returns dataloader pipeline with data augmentation
        """

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        image_datasets = {x: ImageDataset(self[x], transform=data_transforms[x]) for x in data_cat}
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
                       data_cat}
        return dataloaders

    if __name__ == 'main':
        pass
