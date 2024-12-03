import glob
import re
import os.path as osp
import json
from .bases import BaseImageDataset
import os

class CombinedAnimals(BaseImageDataset):
    """
    Combined Animals Dataset for generalizable ReID
    Training: Deer + Hare + Penguin
    Validation: Stoat + Pukeko + Wallaby
    """
    dataset_dir = "CombinedAnimals"

    def __init__(self, root='', verbose=True, pid_begin=0, mode='train'):
        super(CombinedAnimals, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.mode = mode

        # Training animals
        self.deer_dir = osp.join(root, "Deer")
        self.hare_dir = osp.join(root, "Hare")
        self.penguin_dir = osp.join(root, "Penguin")

        # Validation animals 
        self.stoat_dir = osp.join(root, "Stoat")
        self.pukeko_dir = osp.join(root, "Pukeko") 
        self.wallaby_dir = osp.join(root, "Wallaby")

        self._check_before_run()
        self.pid_begin = pid_begin

        if mode == 'train':
            # Combine all training animals data
            train = self._process_training_dir()
            self.train = train
            self.query = []
            self.gallery = []
            self.num_train_pids = self.get_num_pids(train)
        else:
            # Combine validation animals query/gallery
            query = self._process_validation_dir('query')
            gallery = self._process_validation_dir('gallery')  
            self.train = []
            self.query = query
            self.gallery = gallery

        if verbose:
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all required directories exist"""
        if self.mode == 'train':
            for dir_path in [self.deer_dir, self.hare_dir, self.penguin_dir]:
                if not osp.exists(dir_path):
                    raise RuntimeError("{} is not available".format(dir_path))
        else:
            for dir_path in [self.stoat_dir, self.pukeko_dir, self.wallaby_dir]:
                if not osp.exists(dir_path):
                    raise RuntimeError("{} is not available".format(dir_path))

    def _process_dir(self, dir_path):
        """Process one directory"""
        print(f"\nProcessing directory: {dir_path}")
        pattern = re.compile(r'(\d+)_([A-Za-z0-9-]+)_(\d+)')
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths.extend(glob.glob(osp.join(dir_path, '*.JPG')))
        
        print(f"Number of images found: {len(img_paths)}")
        dataset = []
        pid_container = set()
        camid_container = set()
        
        # First pass to collect PIDs and camera IDs
        for img_path in sorted(img_paths):
            basename = osp.basename(img_path)
            match = pattern.match(basename)
            if match:
                pid = int(match.group(1))
                camid = hash(match.group(2)) % 10000
                pid_container.add(pid)
                camid_container.add(camid)
        
        print(f"Found {len(pid_container)} unique PIDs")
        print(f"Found {len(camid_container)} unique camera IDs")
        print(f"PIDs found: {sorted(list(pid_container))}")
        
        # Second pass to build dataset list
        for img_path in sorted(img_paths):
            basename = osp.basename(img_path)
            match = pattern.match(basename)
            if match:
                pid = int(match.group(1))
                camid = hash(match.group(2)) % 10000
                dataset.append((img_path, pid, camid, 0))
        
        print(f"Successfully processed {len(dataset)} images\n")
        return dataset

    def _process_training_dir(self):
        """Process training animals directories"""
        print("\nProcessing training directories...")
        train_data = []
        pid_container = set()
        pid_count = 0  # Debug counter
        
        train_dirs = [
            osp.join(self.deer_dir, 'train'),
            osp.join(self.hare_dir, 'train'),
            osp.join(self.penguin_dir, 'train')
        ]
        
        for animal_dir in train_dirs:
            print(f"\nProcessing training directory: {animal_dir}")
            data = self._process_dir(animal_dir)
            
            # Debug: Print original PIDs
            orig_pids = set([pid for _, pid, _, _ in data])
            print(f"Original PIDs in {os.path.basename(os.path.dirname(animal_dir))}: {sorted(list(orig_pids))}")
            
            # Relabel PIDs for this animal to avoid conflicts
            animal_pid2label = {pid: pid_count + idx for idx, pid in enumerate(sorted(list(orig_pids)))}
            data = [(img_path, animal_pid2label[pid], camid, trackid) 
                    for img_path, pid, camid, trackid in data]
            
            # Debug: Print relabeled PIDs
            new_pids = set([pid for _, pid, _, _ in data])
            print(f"Relabeled PIDs: {sorted(list(new_pids))}")
            
            train_data.extend(data)
            pid_container.update(new_pids)
            pid_count += len(orig_pids)
            
            print(f"Added {len(orig_pids)} new PIDs from {os.path.basename(os.path.dirname(animal_dir))}")
        
        print(f"\nTotal training statistics:")
        print(f"Total images: {len(train_data)}")
        print(f"Total unique PIDs: {len(pid_container)}")
        print(f"All PIDs: {sorted(list(pid_container))}")
        
        return train_data

    def _process_validation_dir(self, mode):
        """Process validation animals directories"""
        print(f"\nProcessing {mode} validation directories...")
        validation_data = []
        
        val_dirs = [
            (self.stoat_dir, 'stoat'),
            (self.pukeko_dir, 'pukeko'),
            (self.wallaby_dir, 'wallaby')
        ]
        
        for animal_dir, animal_name in val_dirs:
            dir_path = osp.join(animal_dir, mode)
            print(f"\nProcessing {mode} directory for {animal_name}: {dir_path}")
            data = self._process_dir(dir_path)
            
            # Keep original PIDs for each species
            # Just add a large offset to avoid any possible overlap
            if animal_name == 'stoat':
                offset = 10000
            elif animal_name == 'pukeko':
                offset = 20000
            else:  # wallaby
                offset = 30000
                
            # Add offset to original PIDs to keep them separate
            data = [(img_path, pid + offset, camid, trackid) 
                    for img_path, pid, camid, trackid in data]
            
            validation_data.extend(data)
            
            # Debug info
            pids = set([pid for _, pid, _, _ in data])
            print(f"Added {len(data)} images from {animal_name}")
            print(f"Number of unique {animal_name} IDs: {len(pids)}")
            print(f"ID range: {min(pids)} to {max(pids)}")
        
        print(f"\nTotal {mode} validation statistics:")
        print(f"Total images: {len(validation_data)}")
        print(f"Total PIDs: {len(set([pid for _, pid, _, _ in validation_data]))}")
        
        return validation_data

    def get_num_pids(self, data):
        """Get number of unique PIDs"""
        pids = set()
        for _, pid, _, _ in data:
            pids.add(pid)
        return len(pids)