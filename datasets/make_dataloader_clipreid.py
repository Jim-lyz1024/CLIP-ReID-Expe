import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .atrw import ATRW
from .friesiancattle2017 import FriesianCattle2017
from .ipanda50 import iPanda50
from .lion import Lion
from .mpdd import MPDD
from .nyala import Nyala
from .polarbear import PolarBear
from .seastar import SeaStar
from .stoat import STOAT

from .penguin import PENGUIN
from .pukeko import PUKEKO
from .hare import HARE
from .deer import DEER
from .wallaby import WALLABY

from .combined_animals import CombinedAnimals

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist

__factory = {
    'atrw' : ATRW, 
    'friesiancattle2017' : FriesianCattle2017, 
    'ipanda50' : iPanda50, 
    'lion' : Lion, 
    'mpdd' : MPDD, 
    'nyala' : Nyala, 
    'polarbear' : PolarBear, 
    'seastar' : SeaStar, 
    'stoat' : STOAT,
    
    'penguin' : PENGUIN,
    'pukeko' : PUKEKO,
    'hare' : HARE,
    'deer' : DEER,
    'wallaby' : WALLABY,
    
    'combined_animals': CombinedAnimals
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    # dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    if cfg.DATASETS.NAMES == 'combined_animals':
        print("\nInitializing training dataset...")
        train_dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, mode='train')
        print("\nInitializing validation dataset...")
        val_dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, mode='val')
        
        train_set = ImageDataset(train_dataset.train, train_transforms)
        train_set_normal = ImageDataset(train_dataset.train, val_transforms)
        val_set = ImageDataset(val_dataset.query + val_dataset.gallery, val_transforms)
        
        num_classes = train_dataset.num_train_pids
        cam_num = train_dataset.num_train_cams
        view_num = train_dataset.num_train_vids
        
        # Create data loaders
        if 'triplet' in cfg.DATALOADER.SAMPLER:
            if cfg.MODEL.DIST_TRAIN:
                print('DIST_TRAIN START')
                mini_batch_size = cfg.SOLVER.STAGE2.IMS_PER_BATCH // dist.get_world_size()
                data_sampler = RandomIdentitySampler_DDP(train_dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
                batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
                train_loader_stage2 = torch.utils.data.DataLoader(
                    train_set,
                    num_workers=num_workers,
                    batch_sampler=batch_sampler,
                    collate_fn=train_collate_fn,
                    pin_memory=True,
                )
            else:
                train_loader_stage2 = DataLoader(
                    train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                    sampler=RandomIdentitySampler(train_dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                    num_workers=num_workers, collate_fn=train_collate_fn
                )
        
        train_loader_stage1 = DataLoader(
            train_set_normal, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True,
            num_workers=num_workers, collate_fn=train_collate_fn
        )
        
        val_loader = DataLoader(
            val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
            num_workers=num_workers, collate_fn=val_collate_fn
        )
        
        return train_loader_stage2, train_loader_stage1, val_loader, len(val_dataset.query), num_classes, cam_num, view_num
    else:
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    
    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.STAGE2.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader_stage2 = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader_stage2 = DataLoader(
                train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader_stage2 = DataLoader(
            train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_stage1 = DataLoader(
        train_set_normal, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=train_collate_fn
    )
    return train_loader_stage2, train_loader_stage1, val_loader, len(dataset.query), num_classes, cam_num, view_num
