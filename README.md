# README

## Data Loading and Transformation Utilities

This set of Python functions provides utility functions for loading image datasets, creating PyTorch dataloaders, and visualizing transformations.

### Requirements

- Python 3.6 or later
- PyTorch
- Matplotlib
- Pillow (PIL)

### Installation

No installation is required, as these are utility functions that can be directly imported and used in your PyTorch projects.

### Usage

#### 1. Walk Through Data Folder

```python
walk_through_datafolder(data_path: str)
```

This function walks through the specified data folder and prints the number of directories and images in each subdirectory.

#### 2. Create Dataloaders

```python
train_dataloader, valid_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir: str,
    valid_dir: str,
    test_dir: str,
    train_transforms: transforms.Compose,
    test_transforms: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS
)
```

This function creates training, validation, and testing dataloaders using PyTorch's `ImageFolder` dataset. It returns a tuple containing the training, validation, and testing dataloaders, along with the class names.

#### 3. Plot Transformed Images

```python
plot_transformed_images(image_paths: str, transform: transforms.Compose, n: int = 3, seed: int = 42)
```

This function plots a series of random images from the specified image paths. It applies the specified transformations and plots the original and transformed images side by side.

#### 4. Plot Random Images from Dataloader

```python
plot_random_images_from_dataloader(dataloader: DataLoader, class_names: list)
```

This function plots a batch of images from the given dataloader, along with their corresponding class names.

### Examples

#### Example 1: Creating Dataloaders

```python
train_dataloader, valid_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir='path/to/train_data',
    valid_dir='path/to/valid_data',
    test_dir='path/to/test_data',
    train_transforms=train_transform,
    test_transforms=test_transform,
    batch_size=32,
    num_workers=4
)
```

#### Example 2: Plotting Transformed Images

```python
plot_transformed_images(image_paths=['path/to/image1.jpg', 'path/to/image2.jpg'],
                        transform=train_transform,
                        n=2,
                        seed=42)
```

#### Example 3: Plotting Random Images from Dataloader

```python
plot_random_images_from_dataloader(dataloader=train_dataloader, class_names=class_names)
```

### License

This code is provided under the MIT License. Feel free to use and modify as needed. Contributions are welcome!