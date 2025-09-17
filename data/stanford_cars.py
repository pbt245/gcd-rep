import os
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy import io as mat_io

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from data.data_utils import subsample_instances

class CarsDataset(Dataset):
    """
        Cars Dataset - Updated for Kaggle Stanford Cars Dataset with nested structure
    """
    def __init__(self, train=True, limit=0, transform=None):
        
        self.loader = default_loader
        self.data = []
        self.target = []
        self.train = train
        self.transform = transform

        # Your specific nested dataset structure
        base_path = "/kaggle/input/stanford-cars-dataset"
        
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Dataset not found at {base_path}")
        
        # Set paths based on your nested structure
        if train:
            data_dir = os.path.join(base_path, "cars_train", "cars_train")  # Nested structure
            metas = os.path.join(base_path, "car_devkit", "devkit", "cars_train_annos.mat")
        else:
            data_dir = os.path.join(base_path, "cars_test", "cars_test")  # Nested structure
            metas = os.path.join(base_path, "car_devkit", "devkit", "cars_test_annos.mat")  # No "withlabels"
        
        if not os.path.exists(metas):
            raise FileNotFoundError(f"Annotation file not found: {metas}")
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Image directory not found: {data_dir}")
        
        print(f"Using annotations: {metas}")
        print(f"Using images: {data_dir}")
        
        # Check how many images are in the directory
        try:
            image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Found {len(image_files)} image files in directory")
        except Exception as e:
            print(f"Error reading image directory: {e}")
        
        # Load annotations
        try:
            labels_meta = mat_io.loadmat(metas)
            print(f"Loaded .mat file with keys: {list(labels_meta.keys())}")
        except Exception as e:
            print(f"Error loading .mat file: {e}")
            raise
        
        # Handle annotations
        annotations = None
        if 'annotations' in labels_meta:
            annotations = labels_meta['annotations'][0]
        else:
            print(f"Available keys in mat file: {list(labels_meta.keys())}")
            raise KeyError("Could not find 'annotations' key in .mat file")

        print(f"Found {len(annotations)} annotations")

        # Process annotations
        successful_loads = 0
        for idx, img_ in enumerate(annotations):
            if limit and idx > limit:
                break
            
            try:
                # Stanford Cars annotation format:
                # img_[0] = bbox_x1, img_[1] = bbox_y1, img_[2] = bbox_x2, img_[3] = bbox_y2
                # img_[4] = class_id, img_[5] = filename
                
                if len(img_) >= 6:
                    img_name = img_[5][0]  # filename
                    class_id = img_[4][0][0]  # class_id
                else:
                    print(f"Unexpected annotation format at index {idx}: length = {len(img_)}")
                    continue
                
                img_path = os.path.join(data_dir, img_name)
                
                # Check if image actually exists
                if os.path.exists(img_path):
                    self.data.append(img_path)
                    self.target.append(class_id)
                    successful_loads += 1
                else:
                    if idx < 5:  # Only print first few missing files to avoid spam
                        print(f"Image not found: {img_path}")
                    
            except (IndexError, AttributeError, TypeError) as e:
                if idx < 5:  # Only print first few errors
                    print(f"Error processing annotation {idx}: {e}")
                continue

        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None
        
        print(f"Successfully loaded {len(self.data)} images out of {len(annotations)} annotations")
        if len(self.data) > 0:
            print(f"Classes range: {min(self.target)} to {max(self.target)} ({len(set(self.target))} unique classes)")
        
        if len(self.data) == 0:
            raise RuntimeError("No images were successfully loaded!")

    def __getitem__(self, idx):
        image = self.loader(self.data[idx])
        target = self.target[idx] - 1  # Convert to 0-based indexing

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]
        return image, target, idx

    def __len__(self):
        return len(self.data)

# Rest of the functions remain the same
def subsample_dataset(dataset, idxs):
    dataset.data = np.array(dataset.data)[idxs].tolist()
    dataset.target = np.array(dataset.target)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]
    return dataset

def subsample_classes(dataset, include_classes=range(98)):  # Changed default to 98
    include_classes_cars = np.array(include_classes) + 1
    cls_idxs = [x for x, t in enumerate(dataset.target) if t in include_classes_cars]
    
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    return dataset

def get_train_val_indices(train_dataset, val_split=0.2):
    train_classes = np.unique(train_dataset.target)
    
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(np.array(train_dataset.target) == cls)[0]
        if len(cls_idxs) > 1:  # Only split if there's more than 1 sample
            v_ = np.random.choice(cls_idxs, replace=False, size=max(1, int(val_split * len(cls_idxs))))
            t_ = [x for x in cls_idxs if x not in v_]
        else:
            t_ = cls_idxs.tolist()
            v_ = []
        
        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs

def get_scars_datasets(train_transform, test_transform, train_classes=range(98), prop_train_labels=0.8,
                    split_train_val=False, seed=0):
    
    np.random.seed(seed)
    
    # Init entire training set
    whole_training_set = CarsDataset(train=True, transform=train_transform)
    
    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes  
    test_dataset = CarsDataset(train=False, transform=test_transform)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets

if __name__ == '__main__':
    # Test loading
    try:
        print("Testing train dataset loading...")
        train_dataset = CarsDataset(train=True)
        print(f"✓ Successfully loaded train dataset with {len(train_dataset)} samples")
        
        print("\nTesting test dataset loading...")
        test_dataset = CarsDataset(train=False) 
        print(f"✓ Successfully loaded test dataset with {len(test_dataset)} samples")
        
        # Test loading a sample
        print("\nTesting sample loading...")
        sample_img, sample_target, sample_idx = train_dataset[0]
        print(f"✓ Successfully loaded sample: target={sample_target}, idx={sample_idx}")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()