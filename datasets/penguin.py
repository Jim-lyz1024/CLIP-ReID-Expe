import glob
import re
import os.path as osp
from .bases import BaseImageDataset

class PENGUIN(BaseImageDataset):
    """
    PENGUIN Dataset
    
    File format:
    animalID_cameraID_number_others.jpg
    e.g. 0_Doc-AIV-DCAMG01_0_00DA4ECE-9A33-4D01-A020-1777D81B0CC7_000001.jpg
    """
    dataset_dir = "Penguin"

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(PENGUIN, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()
        self.pid_begin = pid_begin
        
        # Process each subset
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Penguin loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        # Calculate dataset statistics
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        """Process one directory with image data.
        
        Args:
            dir_path (str): Directory path.
            relabel (bool): If True, relabel the IDs for training set.
        """
        # Get image list
        img_paths = glob.glob(osp.join(dir_path, '*.JPG'))
        img_paths.extend(glob.glob(osp.join(dir_path, '*.jpg')))  # Handle both upper and lower case extensions
        
        # Pattern matches: animalID_cameraID_number_others.jpg
        pattern = re.compile(r'(\d+)_([A-Za-z0-9-]+)_\d+')
        
        # First pass: collect all PIDs and camera IDs
        pid_container = set()
        camid_container = set()
        
        for img_path in sorted(img_paths):
            basename = osp.basename(img_path)
            match = pattern.match(basename)
            if match:
                pid = int(match.group(1))  # Animal ID
                camid = hash(match.group(2)) % 10000  # Hash camera ID to numeric value
                pid_container.add(pid)
                camid_container.add(camid)

        # Generate new labels for training set
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        # Second pass: build dataset list
        dataset = []
        for img_path in sorted(img_paths):
            basename = osp.basename(img_path)
            match = pattern.match(basename)
            if match:
                pid = int(match.group(1))
                camid = hash(match.group(2)) % 10000
                
                # Relabel PIDs for training set
                if relabel:
                    pid = pid2label[pid]
                
                # Final format: (img_path, pid, camid, trackid)
                # trackid is set to 0 as it's not used in this dataset
                dataset.append((img_path, self.pid_begin + pid, camid, 0))

        return dataset