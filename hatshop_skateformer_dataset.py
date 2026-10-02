#!/usr/bin/env python3
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from pathlib import Path
import re
from read_bbox_frames import extract_track_frames
import pickle
import os
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import random



def show_image_grid(images):
    """
    Args:
        images: Tensor of shape [16, 3, 224, 224]
    """

    imgs = images.cpu()

    rows = []
    for r in range(4):
        row = torch.cat([imgs[r * 4 + c] for c in range(4)], dim=2)
        rows.append(row)

    grid = torch.cat(rows, dim=1)

    # [3,H,W] -> [H,W,3]
    grid = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return grid


def temporal_resize(clip, out_len=64):
    # (T,20,3) -> (1,60,T)
    clip = torch.as_tensor(clip, dtype=torch.float32)  # (T,
    x = clip.permute(1, 2, 0).reshape(1, 20 * 3, clip.shape[0])

    # interpolate along time
    x = F.interpolate(
        x,
        size=out_len,
        mode='linear',
        align_corners=False
    )

    # (1,60,64) -> (64,20,3)
    x = x.reshape(20, 3, out_len).permute(2, 0, 1)

    return x


AZURE_TO_NWUCLA = [
    0, 1, 2, 26,
    5, 6, 7, 8,
    12, 13, 14, 15,
    18, 19, 20, 21,
    22, 23, 24, 25,
]

PARTITION_ORDER = [
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
    16, 17, 18, 19,
    1, 2, 0, 3,
]

MOVEMENT_LABELS = {
    "Moving Through the Store": 0,
    "Standing at the Shelf": 1,
    "Standing at the Mirror": 2,
    "Standing at the Counter": 3,
}

ARM_LABELS = {
    "Pick or Place a Hat": 0,
    "Holding Hat": 1,
    "Wear or Take Off a Hat": 2,
    "Idle": 3,
}


def parse_categories(label):
    if isinstance(label, bytes):
        label = label.decode("utf-8")

    if "decision:" in label:
        label = label.split("decision:", 1)[1]

    if "label:" in label:
        label = label.split("label:", 1)[0]

    label = label.strip(" []")
    categories = {}

    for part in label.split(","):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        categories[key.strip()] = value.strip()

    return categories


def find_category(categories, word):
    for key, value in categories.items():
        if word.lower() in key.lower():
            return value
    return None


class HatShopSkateFormerDataset(Dataset):
    def __init__(self, hdf5_paths, target="movement", clip_size = 32, pose_group="/poses_fused", model_yolo = False):
        if isinstance(hdf5_paths, str):
            hdf5_paths = [hdf5_paths]

        if target not in {"movement", "arm"}:
            raise ValueError("target must be 'movement' or 'arm'")

        self.hdf5_paths = hdf5_paths
        self.target = target
        self.pose_group = pose_group
        self.clip_size = clip_size
        self.samples = []
        self.yolo_model = model_yolo
 
        for hdf5_path in self.hdf5_paths:
            self._index_file(hdf5_path)

    def _index_file(self, hdf5_path):
        with h5py.File(hdf5_path, "r") as hdf:
            if self.pose_group not in hdf or "/labels" not in hdf:
                return

            start_time = float(hdf["/metadata"].attrs["bag_start_time"])
            f = open(hdf5_path + ".txt", "w", encoding="utf-8")
            f.write("Dataset Statistics\n")

            video_samples = []
            self.samples = []

            for track_id in hdf["/labels"].keys():
                if track_id not in hdf[self.pose_group]:
                    continue

                labels = hdf[f"/labels/{track_id}"]
                starts = labels["start_times"][:]
                durations = labels["durations"][:]
                actions = labels["actions"][:]
                f.write(f"Track ID: {track_id}, Number of actions: {len(actions)}\n")
                start_times = {}
                end_times = {}
                track_ids = {}
                sample_id = 0
                
                for start, duration, action in zip(starts, durations, actions):
                    categories = parse_categories(action)

                    if self.target == "movement":
                        value = find_category(categories, "Movement")
                        label_map = MOVEMENT_LABELS
                    else:
                        value = find_category(categories, "Arm")
                        label_map = ARM_LABELS

                   # print(value)
                    if value not in label_map:
                        continue
                    f.write(self.target + f" Label: {value}, Start: {start}, Duration: {duration}\n")

                    with h5py.File(hdf5_path, "r") as hdf:
                        group = hdf[f"{self.pose_group}/{track_id}"]
                        timestamps= group["timestamps"][:]
                        joints = group["joints"][:]

                    start_time = float(start)
                    end_time = float(start) + float(duration)
                    mask = (
                        (timestamps >= start_time) &
                        (timestamps < end_time)
                    )

                    joints = joints[mask]
                    timestamps = timestamps[mask]
                    f.write(f"Test is passed: Number of joints: {len(joints)}, Number of timestamps: {len(timestamps)}\n")
                
                    if len(joints) == 0 or len(timestamps) < self.clip_size:
                        continue

                                                       
                    f.write(f"Sample added: Start: {start_time}, End: {end_time}, Label: {label_map[value]}, sample id: {sample_id+1}\n")
                    print(f"label_map = {label_map[value]}\n")
                    self.samples.append({
                        "hdf5_path": hdf5_path,
                        "track_id": track_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "label": label_map[value],
                        "joints": joints,
                        "timestamps": timestamps,
                    })
                    start_times[sample_id] = start_time
                    end_times[sample_id] = end_time
                    track_ids [sample_id] = track_id
                    sample_id = sample_id + 1

            f.close()
             # Reading corresponding bag file and extracting frames
            root_path = Path(hdf5_path).parents[2]
            base_name = re.sub(r"_id\d+", "", Path(hdf5_path).stem)
            bag_path = root_path / "Bags" / f"{base_name}.bag"

            for s in range(len(start_times)):
                start_time = start_times[s]
                print(f"start_time = {start_time}")
                end_time = end_times[s]
                track_id = track_ids[s]

            root_path = "/mnt/data2/output_frames_test/"
            frame_path = "/mnt/data2/original_frames_test/"

            output_dir = os.path.join(root_path, f"{base_name}_track_{track_id}")
            frame_path = os.path.join(frame_path, f"{base_name}_track_{track_id}")

            if True:
                frames, bboxes, im_times = extract_track_frames(
                    bag_path=bag_path,
                    track_ids=track_ids,
                    start_times=start_times,
                    end_times=end_times,
                    yolo_model = self.yolo_model,
                    output_dir = output_dir,
                    frame_dir = frame_path
                )

                video_samples = {
                    "frames": frames,
                    "boxes": bboxes,
                    "im_times":im_times,
                }

                root_path = "/mnt/data2/visualdata_test"
                os.makedirs(root_path, exist_ok=True)

                file_path = os.path.join(root_path, f"{base_name}_track{track_id}.pkl")

                with open(file_path, "wb") as f:
                    pickle.dump(video_samples, f, protocol=pickle.HIGHEST_PROTOCOL)


            root_path2 = "/mnt/data2/skeletondata_test"
            os.makedirs(root_path2, exist_ok=True)
            

            file_path2 = os.path.join(root_path2, f"{base_name}_track{track_id}.pkl")
            skeleton_data = self.samples

            with open(file_path2, "wb") as f:
                pickle.dump(skeleton_data, f, protocol=pickle.HIGHEST_PROTOCOL)


                        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        joints = sample["joints"]
        timestamps = sample["timestamps"]

        joints = joints[:, AZURE_TO_NWUCLA, :]
        joints = joints[:, PARTITION_ORDER, :]


        ns = joints.shape[0]

        start = torch.randint(0, ns - self.clip_size + 1, (1,)).item()
        clip = joints[start:start + self.clip_size]
        clip = temporal_resize(clip)

        # clip: (64, 20, 3)

        root_idx = 0  # replace with your actual root/pelvis joint
        clip = clip - clip[:, root_idx:root_idx+1, :]


        data = clip.permute(2, 0, 1)

        data = data.unsqueeze(-1)

        

        return (
            data,
            sample["label"],
            index,
        )

def crop_and_resize(img, bbox, context_factor=1.75, out_size=224):

    H, W = img.shape[:2]

    if len(bbox)==1:
        bbox = torch.tensor(bbox[0])
    if len(bbox)==2:
            bbox = torch.tensor(bbox[0])
    x1, y1, x2, y2 = bbox
    wb = x2-x1
    hb = y2-y1
    # Center of bbox
    cx = x1 + wb/ 2.0
    cy = y1 + hb / 2.0

    # OSTrack-style scale normalization
   # crop_size = max(wb, hb) * context_factor

    crop_size = np.sqrt(wb*hb+1e-6) * context_factor

    x1 = int(cx - crop_size / 2)
    y1 = int(cy - crop_size / 2)
    x2 = int(cx + crop_size / 2)
    y2 = int(cy + crop_size / 2)

    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - W)
    pad_b = max(0, y2 - H)

    x1_valid = max(0, x1)
    y1_valid = max(0, y1)
    x2_valid = min(W, x2)
    y2_valid = min(H, y2)

    crop = img[y1_valid:y2_valid, x1_valid:x2_valid]

    if pad_l or pad_t or pad_r or pad_b:
        crop = np.pad(
            crop,
            ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
            mode='constant'
        )

    crop_resized = np.array(
    Image.fromarray((crop * 255).astype(np.uint8)).resize(
        (out_size, out_size),
        resample=Image.Resampling.BILINEAR)
        ).astype(np.float32) / 255.0
            
     
    return crop_resized

def extract_person_crop(image, bbox, output_size=224, context_factor=2.0):
        """
        Extract an OSTrack-style square crop around a person.

        Args:
            image (np.ndarray): HxWx3 image.
            bbox (tuple/list): (x, y, w, h).
            output_size (int): Output crop size.
            context_factor (float): Amount of surrounding context.

        Returns:
            crop (np.ndarray): output_size x output_size crop.
        """
        
        x, y, w, h = bbox

        # Center of bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        # OSTrack-style scale normalization
        # crop_size = max(w, h) * context_factor

        crop_size = np.sqrt(w*h+1e-6) * context_factor
        H, W = image.shape[:2]
        image = np.array(image)

        x1 = cx - crop_size / 2
        y1 = cy - crop_size / 2
        x2 = cx + crop_size / 2
        y2 = cy + crop_size / 2

        # Required padding
        left_pad   = max(0, int(np.ceil(-x1)))
        top_pad    = max(0, int(np.ceil(-y1)))
        right_pad  = max(0, int(np.ceil(x2 - W)))
        bottom_pad = max(0, int(np.ceil(y2 - H)))

        if left_pad or top_pad or right_pad or bottom_pad:
            image = cv2.copyMakeBorder(
                image,
                top_pad,
                bottom_pad,
                left_pad,
                right_pad,
                cv2.BORDER_REPLICATE,
            )

        # Shift coordinates after padding
        x1 += left_pad
        x2 += left_pad
        y1 += top_pad
        y2 += top_pad

        crop = image[
            int(round(y1)):int(round(y2)),
            int(round(x1)):int(round(x2))
        ]

        crop = cv2.resize(
            crop,
            (output_size, output_size),
            interpolation=cv2.INTER_LINEAR,
        )

        return crop


def load_and_preprocess_frames(frames, bboxes, device=None):
    """
    Args:
        frames: Dictionary mapping frame indices to image paths.
                Example:
                    {
                        0: "/path/frame_000000.jpg",
                        1: "/path/frame_000001.jpg",
                        ...
                    }

        device: Optional torch device, e.g. "cuda" or "cpu".

    Returns:
        frames_tensor: Tensor of shape [N, 3, H, W],
                       normalized using ImageNet mean/std.
    """

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)

    images = []

    # Sort according to frame index
    for idx in range(len(frames)):
        image_path = frames[idx]
        bbox = bboxes[idx]

        # Read image
        image = Image.open(image_path).convert("RGB")
        image = np.array(Image.open(image_path).convert("RGB")).astype(np.float32) / 255.0

        image_crop = crop_and_resize(image, bbox, context_factor=2.5, out_size=224)

        # Convert [H, W, 3] -> [3, H, W]
        image_crop = torch.from_numpy(image_crop).permute(2, 0, 1)

        images.append(image_crop)

   
    # [N, 3, H, W]
    frames_tensor = torch.stack(images, dim=0)
   # show_image_grid(frames_tensor)

    # Normalize
    frames_tensor = (frames_tensor - mean) / std

    if device is not None:
        frames_tensor = frames_tensor.to(device)

    return frames_tensor


def normalize_filename(filename):
    """
    Normalize filename for matching.

    Examples:
        2024_06_29_s8_merged_tracked.pkl
        -> 2024-06-29_s8_merged_tracked

        2024-06-29_s8_merged_tracked_track0.pkl
        -> 2024-06-29_s8_merged_tracked_track0
    """

    name = os.path.splitext(filename)[0]

    # Normalize date:
    # 2024_06_29 -> 2024-06-29
    name = re.sub(
        r'^(\d{4})_(\d{2})_(\d{2})',
        r'\1-\2-\3',
        name
    )

    return name

import numpy as np


def prepare_frames(frames, bboxes, times, target_frames=16):
    """
    Convert a sequence of <=32 frames to exactly 16 frames.

    - If fewer than 16 frames: repeat/interpolate frames temporally.
    - If 16 or more frames: uniformly sample 16 frames.

    Args:
        frames: List or sequence of frames.
        target_frames: Number of frames required by the model.

    Returns:
        List containing exactly target_frames frames.
    """

    num_frames = len(frames)

    if num_frames == 0:
        raise ValueError("The input contains no frames.")

    # Generate indices uniformly over the original sequence
    indices = np.linspace(
        0,
        num_frames - 1,
        target_frames
    )

    # Round to nearest frame
    indices = np.round(indices).astype(int)
    selected_frames = [frames[i] for i in indices]
    selected_bboxes = [bboxes[i] for i in indices]
    selected_times = [times[i] for i in indices]

    return selected_frames, selected_bboxes, selected_times

import numpy as np


def sample_skeleton_clip(timestamps, start_time, end_time, clip_len=64):
    """
    Args:
        timestamps : 1D array/list of skeleton timestamps (sorted)
        start_time : clip start time
        end_time   : clip end time
        clip_len   : desired output length (default=64)

    Returns:
        sampled_indices : indices into the original skeleton sequence
        start_idx       : closest index to start_time
        end_idx         : closest index to end_time
    """

    timestamps = np.asarray(timestamps)

    # Find closest skeleton frames to visual start/end times
    td = np.min(np.abs(timestamps - start_time))
    if td>0.1:
        print('the time difference is larger than the threshold, check it!!!')
    start_idx = np.argmin(np.abs(timestamps - start_time))
    end_idx = np.argmin(np.abs(timestamps - end_time))

    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    # Candidate skeleton frames within interval
    interval_indices = np.arange(start_idx, end_idx + 1)

    n = len(interval_indices)

    if n == 0:
        raise ValueError("No skeleton frames found in selected interval.")

    if n == clip_len:
        sampled_indices = interval_indices

    else:
        # Works for both n < clip_len and n > clip_len
        positions = np.linspace(0, n - 1, clip_len)
        sampled_indices = interval_indices[np.round(positions).astype(int)]

    return sampled_indices, start_idx, end_idx


class HatShopSkateFormerDatasetPickle(Dataset):
    def __init__(self, video_path, skeleton_path, clip_size = 32):


        self.video_path = video_path
        self.skeleton_path = skeleton_path
        self.clip_size = clip_size
        self.num_cameras = 19
        self.samples = []
         # Get pickle files
        visual_files = [f for f in os.listdir(video_path) if f.endswith(".pkl")]
        skeleton_files = [f for f in os.listdir(skeleton_path) if f.endswith(".pkl")]

         # Use a set for fast lookup
        skeleton_file_set = set(skeleton_files)

        # Sort for reproducibility
        visual_files = sorted(visual_files)

        # Create skeleton lookup
        skeleton_lookup = {}

        for skeleton_file in skeleton_files:
            key = normalize_filename(skeleton_file)
            skeleton_lookup[key] = skeleton_file


        for video_file in sorted(visual_files):

            video_key = normalize_filename(video_file)

            # =========================================================
            # CASE 2:
            # First try exact match INCLUDING track ID
            # =========================================================
            skeleton_file = skeleton_lookup.get(video_key)

            # =========================================================
            # CASE 1:
            # If exact match does not exist, try track0
            # =========================================================
            if skeleton_file is None:
                skeleton_key = video_key + "_track0"
                skeleton_file = skeleton_lookup.get(skeleton_key)

            if skeleton_file is None:
                skeleton_key = video_key + "_track1"
                skeleton_file = skeleton_lookup.get(skeleton_key)

            if skeleton_file is None:
                skeleton_key = video_key + "_track2"
                skeleton_file = skeleton_lookup.get(skeleton_key)
            
            if skeleton_file is None:
                skeleton_key = video_key + "_track3"
                skeleton_file = skeleton_lookup.get(skeleton_key)

            if skeleton_file is None:
                skeleton_key = video_key + "_track300"
                skeleton_file = skeleton_lookup.get(skeleton_key)

            # =========================================================
            # No match
            # =========================================================
            if skeleton_file is None:

                print(
                    f"Warning: No corresponding skeleton file "
                    f"found for {video_file}"
                )

                continue

            print(f"Visual:   {video_file}")
            print(f"Skeleton: {skeleton_file}")

                      
            # Pass both files to the index function
            self._index_file(video_file, skeleton_file)
               
    def _index_file(self, video_file, skeleton_file):



        video_file_path = os.path.join(
            self.video_path,
            video_file
        )

        with open(video_file_path, "rb") as f:
            video_data = pickle.load(f)

        # Read skeleton pickle
        skeleton_file_path = os.path.join(
            self.skeleton_path,
            skeleton_file
        )

        with open(skeleton_file_path, "rb") as f:
            skeleton_data = pickle.load(f)

        # number of samples for the skeleton data
        num_samples = len(skeleton_data)
       
        for s in range(num_samples):
            hdf5_path = skeleton_data[s]['hdf5_path']
            track_id = skeleton_data[s]['track_id']
            label = skeleton_data[s]['label']
            start_time = skeleton_data[s]['start_time']
            end_time = skeleton_data[s]['end_time']
            joints = skeleton_data[s]['joints']
            timestamps = skeleton_data[s]['timestamps']

            frames_all = []
            bboxes_all = []
            frame_times_all = []
            cameras = []
            sample_flag = False
            for i in range(self.num_cameras):
                if video_data['frames'][i]!=[]:
                    frames_path = video_data['frames'][i][s]
                    bboxes = video_data['boxes'][i][s]
                    frame_times = video_data['im_times'][i][s]
                    if len(frames_path)>2:
                        sample_flag = True
                        cameras.append(i)
                        frames_all.append(frames_path)
                        bboxes_all.append(bboxes)
                        frame_times_all.append(frame_times)
                else:
                    continue

            if sample_flag:
                self.samples.append({
                    "hdf5_path": hdf5_path,
                    "track_id": track_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "label": label,
                    "joints": joints,
                    "timestamps": timestamps,
                    "cameras":cameras,
                    "frames":frames_all,
                    "bboxes":bboxes_all,
                    "frame_times":frame_times_all
                })
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        joints = sample["joints"]
        timestamps = sample["timestamps"]
        label =  sample["label"]
        joints = joints[:, AZURE_TO_NWUCLA, :]
        joints = joints[:, PARTITION_ORDER, :]
        frames_all = sample['frames']
        bboxes_all = sample['bboxes']
        frame_times_all = sample['frame_times']

        # Random selection of frames
        nf = len(frames_all)
        if nf == 1:
            frames = frames_all[0]
            bboxes = bboxes_all[0]
            frame_times = frame_times_all[0]
        else:
            n = random.randint(0, nf - 1)
            frames = frames_all[n]
            bboxes = bboxes_all[n]
            frame_times = frame_times_all[n]


        ns_joints = joints.shape[0]
        ns_frames = len(frames)

        # Selection of video frames
        if ns_frames<=32:
            frames_subset = [frames[i] for i in range(ns_frames)]
            bboxes_subset = [bboxes[i] for i in range(ns_frames)]
            frame_times_subset = [frame_times[i] for i in range(ns_frames)]
            clip_video = prepare_frames(frames_subset, bboxes_subset, frame_times_subset, target_frames=16)
        else:
            start = torch.randint(0, ns_frames - self.clip_size + 1, (1,)).item()
            frames_subset = [frames[i] for i in range(start, start + self.clip_size)]
            bboxes_subset = [bboxes[i] for i in range(start, start + self.clip_size)]
            frame_times_subset = [frame_times[i] for i in range(start, start + self.clip_size)]
            clip_video = prepare_frames(frames_subset, bboxes_subset, frame_times_subset, target_frames=16)

      
        frames_tensor = load_and_preprocess_frames(clip_video[0], clip_video[1])
        frames_tensor = frames_tensor.permute(1, 0, 2, 3)
        # frames_tensor (16, 3, 224, 224)

        # Selection of aligned corresponding joints
        start_time = frame_times_subset[0]
        end_time = frame_times_subset[-1]
        sampled_indices, start_idx, end_idx = sample_skeleton_clip(timestamps, start_time, end_time, clip_len=64)
        clip = joints[sampled_indices]
       # clip = temporal_resize(clip)

        # clip: (64, 20, 3)

        root_idx = 0  # replace with your actual root/pelvis joint
        clip = clip - clip[:, root_idx:root_idx+1, :]


        joints_tensor = torch.tensor(clip).permute(2, 0, 1)

        joints_tensor = joints_tensor.unsqueeze(-1)

        

        return (
            frames_tensor,
            joints_tensor,
            label,
            index,
        )
