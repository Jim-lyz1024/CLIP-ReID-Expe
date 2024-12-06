import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from model.make_model import make_model  # Changed to use base model instead of clipreid
from config import cfg
import glob
import re

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transformation
transform = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def extract_id(filename):
    """Extract ID from filename."""
    pattern = r'^(\d+)_'
    match = re.match(pattern, filename)
    if match:
        return int(match.group(1))
    return None

def get_dataset_info(image_dir):
    """Get dataset information including number of unique IDs and cameras."""
    ids = set()
    cameras = set()
    
    for img_path in glob.glob(os.path.join(image_dir, '*.JPG')):
        filename = os.path.basename(img_path)
        id_num = extract_id(filename)
        if id_num is not None:
            ids.add(id_num)
            
        cam_match = re.match(r'^\d+_([^_]+)_', filename)
        if cam_match:
            cameras.add(cam_match.group(1))
    
    return len(ids), len(cameras)

def load_images_and_ids(image_dir):
    """Load images and their corresponding IDs."""
    images = []
    ids = []
    filenames = []
    
    for img_path in sorted(glob.glob(os.path.join(image_dir, '*.JPG'))):
        filename = os.path.basename(img_path)
        id_num = extract_id(filename)
        if id_num is not None:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                images.append(img_tensor)
                ids.append(id_num)
                filenames.append(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return torch.stack(images), np.array(ids), filenames

def main():
    # Load configuration
    cfg.merge_from_file('configs/animal/vit_clipreid.yml')
    cfg.freeze()

    # Get dataset information
    image_dir = '/data/yil708/Code-CLIP-ReID/datasets_meta/TSNE/train/'
    num_classes, num_cameras = get_dataset_info(image_dir)
    print(f"Dataset info - Classes: {num_classes}, Cameras: {num_cameras}")

    # Initialize model with correct parameters
    model = make_model(cfg, num_class=num_classes, camera_num=num_cameras, view_num=1)
    
    # Load the model weights
    state_dict = torch.load('/data/yil708/Code-CLIP-ReID/tests/Hare/CLIP-ReID/ViT-B-16_50.pth')
    
    # Filter out prompt learner related parameters
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                         if not any(x in k for x in ['prompt_learner', 'text_encoder'])}
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Load images and IDs
    print("Loading images...")
    images, ids, filenames = load_images_and_ids(image_dir)

    # Extract features
    print("Extracting features...")
    features = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            # Model forward pass for inference only
            feat = model(batch)  # This will use the model's forward method for inference
            features.append(feat.cpu())
    
    features = torch.cat(features, dim=0).numpy()
    print(f"Feature shape: {features.shape}")

    # Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(images)-1))
    features_2d = tsne.fit_transform(features)

    # Plot
    print("Creating plot...")
    plt.figure(figsize=(12, 8))
    
    # Get unique IDs for color assignment
    unique_ids = np.unique(ids)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_ids)))
    
    # Create scatter plot
    for idx, id_num in enumerate(unique_ids):
        mask = ids == id_num
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[idx]], label=f'ID {id_num}',
                   alpha=0.6, s=100)  # Increased marker size

    plt.title('t-SNE Visualization of Hare ReID Features', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              bbox_transform=plt.gcf().transFigure, 
              fontsize=10)
    
    # Remove axes
    plt.axis('off')
    
    # Save as PDF with high quality
    plt.savefig('hare_tsne_vis.pdf', format='pdf', 
                dpi=300, bbox_inches='tight')
    plt.close()

    print("Visualization saved as 'hare_tsne_vis.pdf'")

if __name__ == '__main__':
    main()