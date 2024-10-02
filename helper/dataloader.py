import zipfile
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from io import BytesIO
import nrrd

class NRRDSliceDataset(Dataset):
    def __init__(self, input_slices, target_slices, labels, transform=None):
        self.input_slices = input_slices
        self.target_slices = target_slices
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.input_slices)

    def __getitem__(self, idx):
        input_slice = self.input_slices[idx]
        target_slice = self.target_slices[idx]
        label = self.labels[idx]

        # Convert slices to tensors and add channel dimension
        input_slice = torch.tensor(input_slice, dtype=torch.float32).unsqueeze(0)
        target_slice = torch.tensor(target_slice, dtype=torch.float32).unsqueeze(0)

        # Apply transformation if provided
        if self.transform:
            input_slice = self.transform(input_slice)
            target_slice = self.transform(target_slice)

        # Normalize input and target slices to [-1, 1] range
        input_slice = (input_slice - input_slice.min()) / (input_slice.max() - input_slice.min())
        input_slice = input_slice * 2 - 1

        target_slice = (target_slice - target_slice.min()) / (target_slice.max() - target_slice.min())
        target_slice = target_slice * 2 - 1

        return input_slice, target_slice, torch.tensor(label, dtype=torch.long)


def load_nrrd_slices_from_zip(zip_file, file_name):
    """Load NRRD slices from a zip archive."""
    with zip_file.open(file_name) as file:
        data = BytesIO(file.read())
        data.seek(0)
        magic_line = data.read(4)
        if magic_line != b'NRRD':
            return []  # Return an empty list to skip invalid files
        data.seek(0)
        header = nrrd.read_header(data)
        data.seek(0)
        volume = nrrd.read_data(header, data)
        slices = [volume[:, :, i] for i in range(40, volume.shape[2] - 40)]
        return slices


def get_dataloaders(zip_files, label_mapping, test_split=0.01, batch_size=1):
    input_slices = []
    target_slices = []
    labels = []

    # Extract and process the files from the zip archives
    for zip_file_path in zip_files:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
            for file_name in zip_file.namelist():
                if file_name.endswith('.nrrd'):
                    if 'spp_me70' in file_name:
                        slices = load_nrrd_slices_from_zip(zip_file, file_name)
                        if slices:
                            input_slices.extend(slices)
                    elif any(kev in file_name for kev in label_mapping.keys()):
                        slices = load_nrrd_slices_from_zip(zip_file, file_name)
                        if slices:
                            target_slices.extend(slices)
                            label_key = [kev for kev in label_mapping.keys() if kev in file_name][0]
                            labels.extend([label_mapping[label_key]] * len(slices))

    # Ensure input slices are repeated to match the target slices
    input_slices = input_slices * (len(target_slices) // len(input_slices))

    # Define the transformations for resizing
    transformations = transforms.Compose([
        transforms.Resize((160, 160), transforms.InterpolationMode.BILINEAR)
    ])

    # Initialize the dataset
    dataset = NRRDSliceDataset(input_slices=input_slices, target_slices=target_slices, labels=labels, transform=transformations)

    # Calculate the number of samples for testing (test_split % of the dataset)
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader objects for both training and testing sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader
