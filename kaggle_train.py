import sys
import os
sys.path.append('/kaggle/input/gcd-files-new')

# Setup
from kaggle_setup import setup_directories, download_dino_weights
setup_directories()
download_dino_weights()

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
import numpy as np
from sklearn.cluster import KMeans

# Import GCD modules
from methods.contrastive_training.contrastive_training import *
from data.get_datasets import get_class_splits
from data.augmentations import get_transform
from project_utils.general_utils import init_experiment
from models import vision_transformer as vits

def main():
    # Kaggle-optimized arguments
    class Args:
        def __init__(self):
            self.batch_size = 32  # Reduced for Kaggle GPU memory
            self.num_workers = 2
            self.eval_funcs = ['v1', 'v2']
            self.dataset_name = 'scars'
            self.prop_train_labels = 0.5
            self.use_ssb_splits = False
            self.grad_from_block = 11
            self.lr = 0.1
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.epochs = 10  # Reduced for faster training
            self.exp_root = '/kaggle/working/experiments'
            self.transform = 'imagenet'
            self.seed = 1
            self.base_model = 'vit_dino'
            self.temperature = 1.0
            self.sup_con_weight = 0.5
            self.n_views = 2
            self.contrast_unlabel_only = False

    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    print(f"Training classes: {len(args.train_classes)}")
    print(f"Unlabeled classes: {len(args.unlabeled_classes)}")
    
    # Initialize experiment
    init_experiment(args, runner_name=['kaggle_gcd'])
    
    # Load DINO model
    if args.base_model == 'vit_dino':
        args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = '/kaggle/working/dino_vitbase16_pretrain.pth'
        
        model = vits.__dict__['vit_base']()
        state_dict = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.to(device)
        
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        args.mlp_out_dim = 65536
        
        # Freeze early layers
        for m in model.parameters():
            m.requires_grad = False
        
        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True
    
    # Get transforms and datasets
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    
    # Get datasets
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(
        args.dataset_name, train_transform, test_transform, args)
    
    print(f"Dataset sizes:")
    print(f"  Labeled train: {len(train_dataset.labelled_dataset)}")
    print(f"  Unlabeled train: {len(train_dataset.unlabelled_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # Create dataloaders with balanced sampling
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))
    
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, 
                             batch_size=args.batch_size, shuffle=False,
                             sampler=sampler, drop_last=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                       batch_size=args.batch_size, shuffle=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                     batch_size=args.batch_size, shuffle=False)
    
    # Projection head
    projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                                               out_dim=args.mlp_out_dim, 
                                               nlayers=args.num_mlp_layers)
    projection_head.to(device)
    
    print(f"Starting training for {args.epochs} epochs...")
    
    # Train
    train(projection_head, model, train_loader, test_loader_labelled, test_loader_unlabelled, args)

if __name__ == "__main__":
    main()