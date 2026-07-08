import os
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from pareconv.utils.registration import get_correspondences
from pareconv.utils.pointcloud import random_sample_rotation, get_transform_from_rotation_translation
import random
import os.path as osp

def read_eth_ply(filename, npoints=None):
    pcd = o3d.io.read_point_cloud(filename)
    points = np.asarray(pcd.points).astype(np.float32)
    if npoints is not None and len(points) >= npoints:
        indices = np.random.choice(len(points), npoints, replace=False)
        points = points[indices]
    return points



class OdometryETHPairDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        point_limit=30000,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1.0,
        return_corr_indices=False,
        matching_radius=0.3,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.point_limit = point_limit

        self.use_augmentation = use_augmentation
        self.augmentation_noise = augmentation_noise
        self.augmentation_rotation = augmentation_rotation

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius

        self.metadata = self._load_metadata()

    def _load_metadata(self):
        scenes = ['gazebo_summer', 'gazebo_winter', 'wood_autmn', 'wood_summer']
        all_pairs = []

        for scene in scenes:
            folder = os.path.join(self.dataset_root, scene)
            pairs, transforms = self._read_transformation_log(folder)
            for (i, (src_id, tgt_id)) in enumerate(pairs):
                data = {
                    'scene': scene,
                    'src_file': os.path.join(folder, f'Hokuyo_{src_id}.ply'),
                    'tgt_file': os.path.join(folder, f'Hokuyo_{tgt_id}.ply'),
                    'transform': transforms[i],
                    'src_frame': src_id,
                    'ref_frame': tgt_id
                }
                all_pairs.append(data)
        return all_pairs


    def _read_transformation_log(self, folder):
        log_path = os.path.join(folder, 'gt.log')
        with open(log_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        pairs, transforms = [], []
        num_pairs = len(lines) // 5

        for i in range(num_pairs):
            line_id = i * 5
            pair_line = lines[line_id].split()
            src_id, tgt_id = int(pair_line[0]), int(pair_line[1])
            pairs.append([src_id, tgt_id])

            transform = []
            for j in range(1, 5):
                transform.append([float(x) for x in lines[line_id + j].split()])
            transforms.append(np.array(transform, dtype=np.float32))

        return np.array(pairs), np.array(transforms)


    def __getitem__(self, index):
        data_dict = {}

        metadata = self.metadata[index]
        data_dict['scene'] = metadata['scene']
        data_dict['ref_frame'] = metadata['ref_frame']
        data_dict['src_frame'] = metadata['src_frame']

        # src_points = read_eth_ply(metadata['src_file'], npoints=self.point_limit)
        # tgt_points = read_eth_ply(metadata['tgt_file'], npoints=self.point_limit)

        src_points = np.array(o3d.io.read_point_cloud(metadata['src_file']).voxel_down_sample(0.3).points)
        tgt_points = np.array(o3d.io.read_point_cloud(metadata['tgt_file']).voxel_down_sample(0.3).points)


        transform = metadata['transform'].copy()

        data_dict['ref_points'] = tgt_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((tgt_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)


        return data_dict

    def __len__(self):
        return len(self.metadata)