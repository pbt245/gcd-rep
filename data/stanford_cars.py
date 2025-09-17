import os
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy import io as mat_io

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from data.data_utils import subsample_instances

class CarsDataset(Dataset):
    """Cars Dataset - Fixed for Kaggle Stanford Cars Dataset"""
    def __init__(self, train=True, limit=0, transform=None, use_train_split=True):
        
        self.loader = default_loader
        self.data = []
        self.target = []
        self.train = train
        self.transform = transform

        base_path = "/kaggle/input/stanford-cars-dataset"
        
        # Always use training data since test has no labels
        data_dir = os.path.join(base_path, "cars_train", "cars_train")
        metas = os.path.join(base_path, "car_devkit", "devkit", "cars_train_annos.mat")
        
        if not os.path.exists(metas) or not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset files not found at {base_path}")
        
        # Load annotations
        labels_meta = mat_io.loadmat(metas)
        annotations = labels_meta['annotations'][0]

        # Process annotations
        for idx, img_ in enumerate(annotations):
            if limit and idx >= limit:
                break
            
            try:
                if len(img_) >= 6:
                    img_name = img_[5][0]
                    class_id = img_[4][0][0]
                    
                    img_path = os.path.join(data_dir, img_name)
                    
                    if os.path.exists(img_path):
                        self.data.append(img_path)
                        self.target.append(class_id)
                    
            except (IndexError, AttributeError, TypeError):
                continue

        self.uq_idxs = np.array(range(len(self.data)))
        self.target_transform = None

    def __getitem__(self, idx):
        image = self.loader(self.data[idx])
        target = self.target[idx] - 1  # Convert to 0-based

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target, self.uq_idxs[idx]

    def __len__(self):
        return len(self.data)

def subsample_dataset(dataset, idxs):
    """Fixed subsample function"""
    if len(idxs) == 0:
        dataset.data = []
        dataset.target = []
        dataset.uq_idxs = np.array([])
        return dataset
    
    max_idx = len(dataset.data) - 1
    valid_idxs = [i for i in idxs if i <= max_idx]
    
    dataset.data = [dataset.data[i] for i in valid_idxs]
    dataset.target = [dataset.target[i] for i in valid_idxs]
    dataset.uq_idxs = np.array(range(len(dataset.data)))
    
    return dataset

def subsample_classes(dataset, include_classes=range(98)):
    include_classes_cars = np.array(include_classes) + 1
    cls_idxs = [x for x, t in enumerate(dataset.target) if t in include_classes_cars]
    
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    return dataset

def get_train_val_test_indices(dataset, train_split=0.6, val_split=0.2):
    """Split dataset indices with stratification"""
    all_classes = np.unique(dataset.target)
    
    train_idxs = []
    val_idxs = []
    test_idxs = []
    
    for cls in all_classes:
        cls_idxs = np.where(np.array(dataset.target) == cls)[0]
        np.random.shuffle(cls_idxs)
        
        n_samples = len(cls_idxs)
        n_train = int(train_split * n_samples)
        n_val = int(val_split * n_samples)
        
        train_idxs.extend(cls_idxs[:n_train])
        val_idxs.extend(cls_idxs[n_train:n_train + n_val])
        test_idxs.extend(cls_idxs[n_train + n_val:])
    
    return train_idxs, val_idxs, test_idxs

def get_train_val_indices(train_dataset, val_split=0.2):
    """Legacy function for compatibility"""
    train_classes = np.unique(train_dataset.target)
    
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(np.array(train_dataset.target) == cls)[0]
        if len(cls_idxs) > 1:
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
    
    # Load full dataset
    full_dataset = CarsDataset(train=True, transform=train_transform, use_train_split=True)
    
    # Split into train/val/test indices
    train_idxs, val_idxs, test_idxs = get_train_val_test_indices(full_dataset)
    
    # Create datasets for each split
    train_dataset = subsample_dataset(deepcopy(full_dataset), train_idxs)
    test_dataset = subsample_dataset(deepcopy(full_dataset), test_idxs)
    test_dataset.transform = test_transform
    
    # Apply class filtering to training data
    whole_training_set = deepcopy(train_dataset)
    
    # Get labelled training set with subsampled classes
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    
    # Subsample some indices from labelled set
    if len(train_dataset_labelled.data) > 0:
        subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
        train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)
    
    # Get unlabelled data
    labelled_original_idxs = set()
    if len(train_dataset_labelled.data) > 0:
        labelled_paths = set(train_dataset_labelled.data)
        for i, path in enumerate(whole_training_set.data):
            if path in labelled_paths:
                labelled_original_idxs.add(i)
    
    all_train_idxs = set(range(len(whole_training_set.data)))
    unlabelled_idxs = list(all_train_idxs - labelled_original_idxs)
    
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), unlabelled_idxs)
    
    # Validation dataset
    val_dataset_labelled = None
    if split_train_val and len(val_idxs) > 0:
        val_dataset_labelled = subsample_dataset(deepcopy(full_dataset), val_idxs)
        val_dataset_labelled.transform = test_transform

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled, 
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets