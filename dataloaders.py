import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import os
from PIL import Image

NUM_WORKERS = os.cpu_count()


def walk_through_datafolder(data_path: str):
    for dirpath, dirnames, filenames in os.walk(data_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath} ")


def create_dataloaders(train_dir: str, valid_dir: str, test_dir: str, train_transforms: transforms.Compose, test_transforms: transforms.Compose, batch_size: int, num_workers: int = NUM_WORKERS):
    """Creates training, validation and testing Dataloders of pytorch.

    Args:
      train_dir (requried): Path to Train data directory
      valid_dir (requried): Path to Validation data directory
      test_dir (requried): Path to Test data directory
      train_transforms (requried): Torchvision transforms for train DataLoader
      test_transforms (requried): Torchvision transforms for valid and test DataLoader
      batch_size: Number of samples per batch in the DataLoader
      num_workers: Number of workers to use in Dataloader (Default is os.cpu_count)

    Returns:
      A tuple of (train_dataloader, validation_dataloader, test_dataloader, class_names).

    Example usuage:
      train_dataloader, valid_dataloader, test_dataloader, class_names = create_dataloaders(
          train_dir: str,
          valid_dir: str,
          test_dir: str,
          train_transforms: transforms.Compose,
          test_transforms: transforms.Compose,
          batch_size: int,
          num_workers: int = NUM_WORKERS
      )                                                                                  )
    """

    # Using ImageFolder to create datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Getting classnames
    class_names = train_data.classes

    # Creating DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    valid_dataloader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Data Loaded Succesfully")
    print(f"Found {len(train_data)} data for Train, {len(valid_data)} data for Validations and {len(test_data)} for Testing")
    print(f"class Names: {class_names}")
    return train_dataloader, test_dataloader, valid_dataloader, class_names


def plot_transformed_images(data_path: str, transform: transforms.Compose, n: int = 3, seed: int = 42):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        data_path (str): Path to data. 
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    image_path_list = list(data_path.glob("*/*/*.jpg"))
    random_image_paths = random.sample(image_path_list, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)


def plot_random_images_from_dataloader(dataloader: DataLoader, class_names):
  """Plots a batch of images in the dataloader.

    Will show the images that will be used in the network

    Args:
        dataloder (Dataloader): Dataloader that we want to see 
        class_names (list): List of class names
    """
  features_batch, labels_batch = next(iter(dataloader))
  batch_size = dataloader.batch_size
  if batch_size > 4:
    cols = rows = 4
  else:
    cols = rows = int(batch_size/2)
  figure = plt.figure(figsize=(8, 8))
  for i in range(1, cols * rows + 1):
    img, label = features_batch[i-1], labels_batch[i-1]
    figure.add_subplot(rows, cols, i)
    plt.title(class_names[label])
    plt.axis("Off");
    plt.imshow(img.squeeze().permute(1,2,0))
  plt.show()