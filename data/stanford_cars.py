import os
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy import io as mat_io

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from data.data_utils import subsample_instances

# Use the global config instead of hardcoded paths
from config import car_root, meta_default_path

class CarsDataset(Dataset):
    """
        Cars Dataset
    """
    def __init__(self, train=True, limit=0, data_dir=None, transform=None, metas=None):
        
        if data_dir is None:
            data_dir = car_root
        if metas is None:
            metas = meta_default_path
            
        data_dir = data_dir.format('train') if train else data_dir.format('test')
        metas = metas.format('train_annos') if train else metas.format('test_annos_withlabels')

        self.loader = default_loader
        self.data_dir = data_dir
        self.data = []
        self.target = []
        self.train = train

        self.transform = transform

        if not isinstance(metas, str):
            raise Exception("Train metas must be string location !")
        
        # Check if files exist, if not try alternative paths
        if not os.path.exists(metas):
            # Try alternative Kaggle dataset structure
            alt_path = "/kaggle/input/stanford-cars-dataset/"
            if train:
                metas = os.path.join(alt_path, "cars_train_annos.mat")
                data_dir = os.path.join(alt_path, "cars_train/")
            else:
                metas = os.path.join(alt_path, "cars_test_annos_withlabels.mat")
                data_dir = os.path.join(alt_path, "cars_test/")
        
        labels_meta = mat_io.loadmat(metas)

        for idx, img_ in enumerate(labels_meta['annotations'][0]):
            if limit:
                if idx > limit:
                    break

            self.data.append(os.path.join(data_dir, img_[5][0]))
            self.target.append(img_[4][0][0])

        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None

    def __getitem__(self, idx):
        image = self.loader(self.data[idx])
        target = self.target[idx] - 1

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]

        return image, target, idx

    def __len__(self):
        return len(self.data)

# Rest of the functions remain the same...
def subsample_dataset(dataset, idxs):
    dataset.data = np.array(dataset.data)[idxs].tolist()
    dataset.target = np.array(dataset.target)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]
    return dataset

def subsample_classes(dataset, include_classes=range(160)):
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
        cls_idxs = np.where(train_dataset.target == cls)[0]
        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]
        
        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs

def get_scars_datasets(train_transform, test_transform, train_classes=range(160), prop_train_labels=0.8,
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