import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from model.make_model import make_model
from config import cfg
import glob
import re
import pickle
from datetime import datetime
import hashlib

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transformation
transform = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_model_hash(model_path):
    """Generate a hash for the model file to ensure cache validity."""
    with open(model_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:8]

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

def extract_and_save_features(model, image_dir, cache_file, model_path):
    """Extract features from all images and save to cache file."""
    images = []
    ids = []
    filenames = []
    
    # First collect all valid images and their info
    print("Collecting image information...")
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
    
    images = torch.stack(images)
    ids = np.array(ids)
    
    # Extract features in batches
    print("Extracting features...")
    features = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            feat = model(batch)
            features.append(feat.cpu())
            if (i + batch_size) % 100 == 0:
                print(f"Processed {i + batch_size}/{len(images)} images")
    
    features = torch.cat(features, dim=0).numpy()
    
    # Save to cache file
    cache_data = {
        'features': features,
        'ids': ids,
        'filenames': filenames,
        'extraction_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_hash': get_model_hash(model_path)
    }
    
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"Features saved to {cache_file}")
    return features, ids, filenames

def load_cached_features(cache_file, model_path):
    """Load features from cache file if valid."""
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        current_model_hash = get_model_hash(model_path)
        if cache_data['model_hash'] != current_model_hash:
            print("Model has changed since feature extraction, cache invalid")
            return None, None, None
        
        print(f"Loaded cached features extracted on {cache_data['extraction_date']}")
        return cache_data['features'], cache_data['ids'], cache_data['filenames']
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None, None, None

def create_tsne_visualization(features, ids, save_path='hare_tsne_vis.pdf'):
    """Create and save TSNE visualization."""
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
    features_2d = tsne.fit_transform(features)

    print("Creating plot...")
    plt.figure(figsize=(12, 8))
    
    # Define a high-contrast color palette for 31 IDs
    color_palette = [
        '#065143',  # green
        '#00FF00',  # Lime
        '#0000FF',  # Blue
        '#FF00FF',  # Magenta
        '#483D03',
        '#98473E',  # Orange
        '#800080',  # Purple
        '#008000',  # Green
        '#79807F',  # 8
        '#4B0082',  # Indigo
        '#985F6F',  # Gold
        '#000080',  # Navy
        '#FF69B4',  # HotPink
        '#3CBBB1',  # Teal
        '#7CCD7C',  # PaleGreen
        '#9370DB',  # MediumPurple
        '#CD5C5C',  # IndianRed
        '#4169E1',  # RoyalBlue
        '#8B4513',  # SaddleBrown
        '#FA8072',  # Salmon
        '#48D1CC',  # MediumTurquoise
        '#C71585',  # MediumVioletRed
        '#DEB887',  # BurlyWood
        '#B66D0D',  # 40
        '#FF8C00',  # DarkOrange
        '#9932CC',  # DarkOrchid
        '#8FBC8F',  # DarkSeaGreen
        '#E9967A',  # DarkSalmon
        '#8A2BE2',  # BlueViolet
        '#2F4F4F',  # DarkSlateGray
        '#DAA520',  # GoldenRod
    ]

    unique_ids = np.unique(ids)
    for idx, id_num in enumerate(unique_ids):
        mask = ids == id_num
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=[color_palette[idx]], label=f'ID {id_num}',
                   alpha=0.8, s=100)

    plt.title('t-SNE Visualization of Hare ReID Features', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              bbox_transform=plt.gcf().transFigure, 
              fontsize=10)
    plt.axis('off')
    
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved as '{save_path}'")

def main():
    # Paths
    image_dir = '/data/yil708/Code-CLIP-ReID/datasets_meta/TSNE/train/'
    model_path = '/data/yil708/Code-CLIP-ReID/tests/Hare/CLIP-ReID/ViT-B-16_50.pth'
    cache_dir = './feature_cache'
    cache_file = os.path.join(cache_dir, f'hare_features_cache.pkl')

    # Load configuration
    cfg.merge_from_file('configs/animal/vit_clipreid.yml')
    cfg.freeze()

    # Get dataset information
    num_classes, num_cameras = get_dataset_info(image_dir)
    print(f"Dataset info - Classes: {num_classes}, Cameras: {num_cameras}")

    # Try to load cached features first
    features, ids, filenames = load_cached_features(cache_file, model_path)

    if features is None:
        # Initialize model
        model = make_model(cfg, num_class=num_classes, camera_num=num_cameras, view_num=1)
        state_dict = torch.load(model_path)
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                             if not any(x in k for x in ['prompt_learner', 'text_encoder'])}
        model.load_state_dict(filtered_state_dict, strict=False)
        model = model.to(device)
        model.eval()

        # Extract and cache features
        features, ids, filenames = extract_and_save_features(model, image_dir, cache_file, model_path)

    # Create visualization
    create_tsne_visualization(features, ids)

if __name__ == '__main__':
    main()