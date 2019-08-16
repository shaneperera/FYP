import os
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader

# Study level data pipeline

# Define the data categories (Training or testing)
data_cat = ['train', 'valid']


# Create function to create dictionary regarding pathway to specific study type, number of patients in study type and
def get_image_data(study_type):
    """
    Returns a dictionary, with keys 'train' and 'valid' and respective values as study level data frames, these data
    frames contain three columns:
         Path: Path directory to data
         Count: Total number of studies for that patient (eg. can have multiple positive or negative case)
         Label: 0 or 1 to represent positive classification or not
    Inputs:
          Study_type (string): one of the seven study type folder names in 'train/valid' dataset
    """

    # Define empty dictionary
    image_data = {}

    # Define study classification dictionary --> Reference key to determine classification
    study_label = {'positive': 1, 'negative': 0}

    # Populate dictionary based on whether it is the valid/train data set
    # Example of directory to MURA data set -->  C:\Users\shann\Documents\FYP\MURA-v1.1\train\...
    for category in data_cat:

        # Locate the MURA dataset
        # Include directory extension --> Special type of string formatting
        base_directory = 'C:/Users/shann/Documents/FYP/MURA-v1.1/%s/%s/' % (category, study_type)

        # List of all the patients inside the study level
        # os.walk --> Generates file names in the directory --> When you enter XR_ELBOW --> Will print directory first
        # [0][1] Indexes the first element, but second element of the first element1
        patients = list(os.walk(base_directory))[0][1]

        # Each category (train/valid) will have 3 columns (the key is based on the path directory)
        # NOTE: There is a unique index as well --> Entries are path,count,label
        image_data[category] = pd.DataFrame(columns=['Path', 'Label'])

        # Populate the rows (path,count,label) of the pandas dataframe
        i = 0  # Used to index the rows of the pandas data frame (table)
        for patient in tqdm(patients):  # Create a loading bar when populating the table
            # os.listdir --> Returns a list of files in the directory
            # (difference with os.walk is that you can't specify the direction)
            for study in os.listdir(base_directory + patient):
                # We need image level support
                for image in os.listdir(base_directory + patient + '/' + study):
                    # Chance of study1_negative & study2_positive (for eg.)
                    path = base_directory + patient + '/' + study + '/' + image
                    label = study_label[study.split('_')[1]]  # Notation: study1_negative -> Splits --> Choose term after _
                    image_data[category].loc[i] = [path, label]  # add new row to data frame
                    # .loc gets rows from particular labels (eg. patient number)
                    i += 1
    return image_data


class ImageDataset(Dataset):
    """Training dataset."""
    # This class will use the dictionary defined above (study_data) to feed an image into the network
    def __init__(self, df, transform=None):
        """
        Args:
            df = Pandas data frame with image path,count and labels (pd.DataFrame)
            transform (callable, optional): Optional transform to be applied on a sample x-ray.
        """

        self.df = df
        self.transform = transform  # Transforms include rotations / scaling / etc

    def __len__(self):
        # Returns the length of dataframe
        return len(self.df)

    def __getitem__(self, idx):
        # Returns an image based on the index (idx) in the pandas dataframe
        # Pandas dataframe has notation: Path, Count, Label
        # iloc -> Identifies which rows (idx) and which columns (0)

        img_path = self.df.iloc[idx, 0]
        image = pil_loader(img_path)
        label = self.df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        # Create a dictionary which holds all the transformed images in a single list (original isn't fed into dict)
        # Label --> Classification of positive or negative
        sample = {'images': image, 'label': label}

        return sample

# def my_collate(batch):
#     data = [item['images'] for item in batch]  #  form a list of tensor
#     target = [item['label'] for item in batch]
#     target = torch.LongTensor(target)
#     return [data, target]

def get_dataloaders(data, batch_size):
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
    # For each image in the category apply the same transformations
    image_datasets = {x: ImageDataset(data[x], transform=data_transforms[x]) for x in data_cat}

    # Load in batches of 8 images into the Neural Network
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=8) for x in
                   data_cat}
    return dataloaders


if __name__ == 'main':  # When this is run alone --> Should return null operation --> Must be imported in main fnc
    pass
