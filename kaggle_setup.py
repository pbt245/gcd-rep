import torch
import os
import urllib.request

def download_dino_weights():
    """Download DINO pretrained weights if not exists"""
    dino_path = '/kaggle/working/dino_vitbase16_pretrain.pth'
    if not os.path.exists(dino_path):
        print("Downloading DINO pretrained weights...")
        url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        urllib.request.urlretrieve(url, dino_path)
        print("DINO weights downloaded successfully!")
    return dino_path

def setup_directories():
    """Create necessary directories"""
    dirs = [
        '/kaggle/working/experiments',
        '/kaggle/working/extracted_features',
        '/kaggle/working/ssb_splits'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Directories created successfully!")

if __name__ == "__main__":
    setup_directories()
    download_dino_weights()