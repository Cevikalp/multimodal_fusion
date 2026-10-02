
import pickle
import random
import torch
from torch.utils.data import Dataset
import os
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
    #show_image_grid(frames_tensor)

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
   # if td>0.1:
    #    print('the time difference is larger than the threshold, check it!!!')
    start_idx = np.argmin(np.abs(timestamps - start_time))
    end_idx = np.argmin(np.abs(timestamps - end_time))

    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    # Candidate skeleton frames within interval
    interval_indices = np.arange(start_idx, end_idx + 1)

    n = len(interval_indices)
    #if n>clip_len:
     #   n = clip_len

    if n == 0:
        raise ValueError("No skeleton frames found in selected interval.")

    if n == clip_len:
        sampled_indices = interval_indices[0:clip_len]

    else:
        # Works for both n < clip_len and n > clip_len
        positions = np.linspace(0, n - 1, clip_len)
        sampled_indices = interval_indices[np.round(positions).astype(int)]

    return sampled_indices, start_idx, end_idx



class HatShopSkateFormerDatasetPickleMultiple(Dataset):

    def __init__(
        self,
        video_path,
        skeleton_path,
        clip_size=16,
        target_frames=16,
        target_fps=12.0,
    ):
        """
        Multi-camera HatShop dataset.

        Each sample may contain a different number of available cameras,
        but there are 19 possible camera IDs in total.

        Video sampling:
            1. Find temporal interval common to all available cameras
               and the annotated skeleton interval.
            2. Randomly select a temporal window.
            3. Generate 32 common target timestamps.
            4. For every camera, select the frame closest to each
               target timestamp.
            5. Convert the 32 synchronized frames to 16 frames using
               prepare_frames().
            6. Store the result in its original camera slot.

        Output:
            frames_tensor:
                [19, 16, 3, 224, 224]

            joints_tensor:
                [3, 64, 20, 1]

            label:
                class label

            camera_mask:
                [19]
                True  -> camera is available
                False -> camera is missing

            index:
                dataset index

        IMPORTANT:
            frame_times for different cameras are assumed to be on
            the same synchronized time axis.
        """

        self.video_path = video_path
        self.skeleton_path = skeleton_path

        # ---------------------------------------------------------
        # Video sampling parameters
        # ---------------------------------------------------------
        self.clip_size = clip_size
        self.target_frames = target_frames
        self.target_fps = target_fps

        # ---------------------------------------------------------
        # TOTAL POSSIBLE CAMERAS
        # ---------------------------------------------------------
        self.num_cameras = 19

        self.samples = []

        # =========================================================
        # GET PICKLE FILES
        # =========================================================

        visual_files = [f for f in os.listdir(video_path) if f.endswith(".pkl")]
        skeleton_files = [f for f in os.listdir(skeleton_path) if f.endswith(".pkl")]
        
        # =========================================================
        # CREATE SKELETON LOOKUP
        # =========================================================

        skeleton_lookup = {}

        for skeleton_file in skeleton_files:

            key = normalize_filename(skeleton_file)
            skeleton_lookup[key] = skeleton_file

        # =========================================================
        # SORT FOR REPRODUCIBILITY
        # =========================================================

        visual_files = sorted(visual_files)

        # =========================================================
        # MATCH VISUAL AND SKELETON FILES
        # =========================================================

        for video_file in visual_files:

            video_key = normalize_filename(
                video_file
            )
            # -----------------------------------------------------
            # First try exact match INCLUDING track ID
            # -----------------------------------------------------
            skeleton_file = skeleton_lookup.get(
                video_key
            )
            # -----------------------------------------------------
            # If exact match does not exist, try track IDs
            # -----------------------------------------------------

            if skeleton_file is None:

                possible_keys = [
                    video_key + "_track0",
                    video_key + "_track1",
                    video_key + "_track2",
                    video_key + "_track3",
                    video_key + "_track300",
                ]

                for skeleton_key in possible_keys:

                    skeleton_file = skeleton_lookup.get(
                        skeleton_key
                    )

                    if skeleton_file is not None:
                        break

            # -----------------------------------------------------
            # No matching skeleton file
            # -----------------------------------------------------

            if skeleton_file is None:

                print(
                    f"Warning: No corresponding skeleton file "
                    f"found for {video_file}"
                )
                continue

            print(f"Visual:   {video_file}")

            print(f"Skeleton: {skeleton_file}")

            # -----------------------------------------------------
            # Index this pair
            # -----------------------------------------------------

            self._index_file(video_file, skeleton_file)

    # =============================================================
    # INDEX ONE VISUAL/SKELETON FILE PAIR
    # =============================================================

    def _index_file(self, video_file, skeleton_file):

        # =========================================================
        # READ VISUAL PICKLE
        # =========================================================

        video_file_path = os.path.join(
            self.video_path,
            video_file
        )

        with open(video_file_path, "rb") as f:
            video_data = pickle.load(f)
        # =========================================================
        # READ SKELETON PICKLE
        # =========================================================

        skeleton_file_path = os.path.join(
            self.skeleton_path,
            skeleton_file
        )

        with open(skeleton_file_path, "rb") as f:
            skeleton_data = pickle.load(f)

        # =========================================================
        # NUMBER OF SKELETON SAMPLES
        # =========================================================

        num_samples = len(skeleton_data)

        # =========================================================
        # INDEX EACH SAMPLE
        # =========================================================

        for s in range(num_samples):

            hdf5_path = skeleton_data[s]['hdf5_path']
            track_id = skeleton_data[s]['track_id']
            label = skeleton_data[s]['label']
            start_time = skeleton_data[s]['start_time']
            end_time = skeleton_data[s]['end_time']
            joints = skeleton_data[s]['joints']
            timestamps = skeleton_data[s]['timestamps']

            # -----------------------------------------------------
            # Containers for available cameras
            # -----------------------------------------------------
            frames_all = []
            bboxes_all = []
            frame_times_all = []
            cameras = []

            # =====================================================
            # COLLECT ALL AVAILABLE CAMERAS
            # =====================================================

            for camera_id in range(self.num_cameras):

                # -------------------------------------------------
                # Make sure camera exists
                # -------------------------------------------------

                if camera_id >= len(
                    video_data["frames"]
                ):
                    continue

                camera_frames = video_data["frames"][camera_id]

                # -------------------------------------------------
                # Camera is completely absent
                # -------------------------------------------------

                if camera_frames == []:
                    continue

                # -------------------------------------------------
                # Make sure sample index exists
                # -------------------------------------------------

                if s >= len(
                    camera_frames
                ):
                    continue

                frames_path = camera_frames[s]

                bboxes = video_data["boxes"][camera_id][s]

                frame_times = video_data["im_times"][camera_id][s]

                # -------------------------------------------------
                # Skip cameras with too few frames
                # -------------------------------------------------

                if len(frames_path) <= 2:
                    continue

                # -------------------------------------------------
                # Make sure frame/time lengths agree
                # -------------------------------------------------

                if len(frame_times) != len(frames_path):

                    print(
                        f"Warning: frame/time mismatch "
                        f"in {video_file}, "
                        f"sample {s}, "
                        f"camera {camera_id}"
                    )

                    continue

                # -------------------------------------------------
                # Store camera information
                #
                # IMPORTANT:
                # cameras contains the ORIGINAL camera ID.
                # -------------------------------------------------

                cameras.append(camera_id)

                frames_all.append(frames_path)

                bboxes_all.append(bboxes)

                frame_times_all.append(frame_times)

            # =====================================================
            # STORE SAMPLE IF AT LEAST ONE CAMERA EXISTS
            # =====================================================

            if len(cameras) > 0:

                self.samples.append(
                    {
                        "hdf5_path": hdf5_path,
                        "track_id": track_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "label": label,
                        "joints": joints,
                        "timestamps": timestamps,
                        "cameras": cameras,
                        "frames": frames_all,
                        "bboxes": bboxes_all,
                        "frame_times": frame_times_all,
                    }
                )

    # =============================================================
    # LENGTH
    # =============================================================

    def __len__(self):

        return len(self.samples)

    # =============================================================
    # FIND COMMON TEMPORAL INTERVAL
    # =============================================================

    def _get_common_time_interval(
        self,
        frame_times_all,
        annotation_start,
        annotation_end,
    ):
        """
        Find the temporal interval common to:

            - all available cameras
            - the annotated action interval

        Assumes frame_times are timestamps on a common
        synchronized timeline.
        """

        camera_starts = []
        camera_ends = []

        # ---------------------------------------------------------
        # Determine temporal coverage of every camera
        # ---------------------------------------------------------

        for frame_times in frame_times_all:

            if len(frame_times) == 0:
                continue

            # Convert only for finding the interval
            times = torch.as_tensor(list(frame_times.values()), dtype=torch.float64)

            camera_starts.append(
                float(times.min())
            )

            camera_ends.append(
                float(times.max())
            )

        if len(camera_starts) == 0:

            return None, None

        # ---------------------------------------------------------
        # Intersection of all camera intervals
        # ---------------------------------------------------------

        common_start = max(camera_starts)

        common_end = min(camera_ends)

        # ---------------------------------------------------------
        # Also restrict to the annotated interval
        # ---------------------------------------------------------

        common_start = max(common_start, float(annotation_start))

        common_end = min(common_end, float(annotation_end))

        # ---------------------------------------------------------
        # Invalid interval
        # ---------------------------------------------------------

        if common_end <= common_start:

            return None, None

        return (
            common_start,
            common_end
        )

    # =============================================================
    # SELECT RANDOM TEMPORAL WINDOW
    # =============================================================

    def _select_temporal_window(
        self,
        common_start,
        common_end,
    ):
        """
        Select a random temporal window.

        The old pipeline effectively used approximately 12 FPS
        after selecting every second frame from a ~24 FPS video.

        For 32 samples, the temporal span is:

            (32 - 1) / 12 = 2.5833 seconds

        rather than 32 / 12 because 32 timestamps contain
        31 intervals.
        """

        if self.clip_size <= 1:

            return (
                common_start,
                common_start
            )

        # ---------------------------------------------------------
        # Duration represented by clip_size samples
        # ---------------------------------------------------------

        clip_duration = (float((2*self.clip_size) - 1)/ float(self.target_fps))

        available_duration = (common_end - common_start)

        # ---------------------------------------------------------
        # If available interval is shorter than desired interval,
        # use the complete available interval.
        # ---------------------------------------------------------

        if available_duration <= clip_duration:

            return (
                common_start,
                common_end
            )

        # ---------------------------------------------------------
        # Random starting time
        # ---------------------------------------------------------

        max_start = (common_end - clip_duration)

        clip_start = random.uniform(common_start, max_start)

        clip_end = (clip_start+ clip_duration)

        return (clip_start, clip_end)

    # =============================================================
    # SAMPLE FRAMES ACCORDING TO TIMESTAMPS
    # =============================================================

    @staticmethod
    def _sample_frames_by_time(
        frames,
        bboxes,
        frame_times,
        target_times,
    ):
        """
        Select the frame closest to every target timestamp.

        Example:

            target timestamp = 1.250 sec

        If a camera has:

            1.21
            1.25
            1.29

        the frame at 1.25 is selected.

        The operation is performed independently for each camera,
        but ALL cameras use the same target_times.
        """
        frame_times_tensor = torch.as_tensor(list(frame_times.values()),dtype=torch.float64)
       
        target_times_tensor = torch.as_tensor(target_times, dtype=torch.float64)

        # ---------------------------------------------------------
        # Make sure timestamps are sorted.
        #
        # Normally they should already be sorted. If they are not,
        # sorting here would require reordering frames and bboxes
        # as well, so we explicitly check instead.
        # ---------------------------------------------------------

        if len(frame_times_tensor) > 1:

            if torch.any(
                frame_times_tensor[1:]
                < frame_times_tensor[:-1]
            ):

                raise RuntimeError(
                    "frame_times must be sorted "
                    "in ascending order."
                )

        sampled_frames = []
        sampled_bboxes = []
        sampled_times = []

        # =========================================================
        # SELECT CLOSEST FRAME FOR EACH TARGET TIME
        # =========================================================

        for target_time in target_times_tensor:

            # -----------------------------------------------------
            # Difference between every available frame and
            # target timestamp
            # -----------------------------------------------------

            distances = torch.abs(frame_times_tensor- target_time)

            frame_idx = torch.argmin(distances).item()

            sampled_frames.append(frames[frame_idx])

            sampled_bboxes.append(bboxes[frame_idx])

            sampled_times.append(frame_times[frame_idx])

        return (
            sampled_frames,
            sampled_bboxes,
            sampled_times,
        )

    # =============================================================
    # GET ITEM
    # =============================================================

    def __getitem__(
        self,
        index
    ):

        sample = self.samples[index]

        # =========================================================
        # SKELETON DATA
        # =========================================================

        joints = sample["joints"]

        timestamps = sample["timestamps"]

        label = sample["label"]

        #annotation_start = sample["start_time"]
        annotation_start = timestamps[0]
        #annotation_end = sample["end_time"]
        annotation_end = timestamps[-1]

        # =========================================================
        # MAP AZURE 32 JOINTS -> NW-UCLA 20 JOINTS
        # =========================================================

        joints = joints[:,AZURE_TO_NWUCLA,:]

        joints = joints[:,PARTITION_ORDER,:]

        # =========================================================
        # VIDEO DATA
        # =========================================================

        frames_all = sample["frames"]

        bboxes_all = sample["bboxes"]

        frame_times_all = sample["frame_times"]

        cameras = sample["cameras"]

        # Number of available cameras for this sample
        nf = len(frames_all)

        if nf == 0:
            raise RuntimeError(
                f"No cameras available for sample "
                f"{index}."
            )

        # =========================================================
        # 1. FIND COMMON TEMPORAL INTERVAL
        # =========================================================

        common_start, common_end = self._get_common_time_interval(frame_times_all, annotation_start, annotation_end)

        if common_start is None:

            raise RuntimeError(
                f"No common temporal interval for "
                f"sample {index}. "
                f"Camera IDs: {cameras}"
            )

        # =========================================================
        # 2. SELECT RANDOM TEMPORAL WINDOW
        # =========================================================

        clip_start_time, clip_end_time = self._select_temporal_window(common_start, common_end)

        # =========================================================
        # 3. CREATE COMMON TARGET TIMESTAMPS
        # =========================================================

        #
        # IMPORTANT:
        #
        # Every camera receives exactly the same target timestamps.
        #
        # Example:
        #
        # Camera 0: 24 FPS
        # Camera 5: 20 FPS
        # Camera 12: 15 FPS
        #
        # They will all be sampled at:
        #
        # t0, t1, t2, ..., t31
        #
        # using the closest available frame.
        #

        target_times = torch.linspace(
            clip_start_time,
            clip_end_time,
            steps=self.clip_size,
            dtype=torch.float64,
        )

        # =========================================================
        # 4. CREATE EMPTY 19-CAMERA OUTPUT
        # =========================================================

        camera_video_tensors = {}

        # =========================================================
        # 5. PROCESS EVERY AVAILABLE CAMERA
        # =========================================================

        for c in range(nf):

            # -----------------------------------------------------
            # IMPORTANT:
            #
            # c is NOT the camera ID.
            #
            # cameras[c] contains the original camera ID.
            # -----------------------------------------------------

            camera_id = cameras[c]
            frames = frames_all[c]
            bboxes = bboxes_all[c]
            frame_times = frame_times_all[c]

            # =====================================================
            # SAMPLE ACCORDING TO COMMON TIME GRID
            # =====================================================

            frames_subset, bboxes_subset, frame_times_subset = self._sample_frames_by_time(frames, bboxes, frame_times, target_times)

            # =====================================================
            # 32 SYNCHRONIZED FRAMES -> 16 FRAMES
            # =====================================================

            clip_video = prepare_frames(frames_subset, bboxes_subset, frame_times_subset, target_frames=self.target_frames)
            
            # =====================================================
            # LOAD AND PREPROCESS
            # =====================================================

            camera_tensor = load_and_preprocess_frames(clip_video[0], clip_video[1])

            # -----------------------------------------------------
            if camera_tensor.ndim != 4:

                raise RuntimeError(
                    f"Unexpected camera tensor shape: "
                    f"{camera_tensor.shape}"
                )

            # -----------------------------------------------------
           
            camera_tensor = camera_tensor.permute(1, 0, 2, 3)

            # -----------------------------------------------------
            # Store using ORIGINAL camera ID
            # -----------------------------------------------------

            camera_video_tensors[camera_id] = camera_tensor

        # =========================================================
        # 6. CREATE FIXED 19-CAMERA TENSOR
        # =========================================================

        #
        # All samples now have:
        #
        #     [19, 16, 3, 224, 224]
        #
        # Missing cameras remain zero.
        #

        if len(camera_video_tensors) == 0:

            raise RuntimeError(
                f"No valid camera data for sample "
                f"{index}."
            )

        # ---------------------------------------------------------
        # Get tensor dimensions from first available camera
        # ---------------------------------------------------------

        first_camera_tensor = next(iter(camera_video_tensors.values()))

        num_frames = (first_camera_tensor.shape[0])

        num_channels = (first_camera_tensor.shape[1])

        height = (first_camera_tensor.shape[2])

        width = (first_camera_tensor.shape[3])

        # ---------------------------------------------------------
        # Allocate all 19 camera slots
        # ---------------------------------------------------------

        frames_tensor = torch.zeros(
            (
                self.num_cameras,
                num_frames,
                num_channels,
                height,
                width,
            ),
            dtype=first_camera_tensor.dtype,
        )

        # =========================================================
        # 7. CAMERA MASK
        # =========================================================

        #
        # True  = camera exists
        # False = camera missing
        #

        camera_mask = torch.zeros(
            self.num_cameras,
            dtype=torch.bool
        )

        # =========================================================
        # 8. INSERT AVAILABLE CAMERAS INTO THEIR ORIGINAL SLOTS
        # =========================================================

        for camera_id, camera_tensor in camera_video_tensors.items():

            frames_tensor[camera_id] = camera_tensor

            camera_mask[camera_id] = True

        # =========================================================
        # 9. SAMPLE ALIGNED SKELETON
        # =========================================================

        #
        # IMPORTANT:
        #
        # Skeleton is sampled using exactly the same temporal
        # interval selected for the synchronized video cameras.
        #

        sampled_indices, start_idx, end_idx = sample_skeleton_clip(timestamps, clip_start_time, clip_end_time, clip_len=64)

        clip = joints[sampled_indices]

        # =========================================================
        # 10. ROOT CENTERING
        # =========================================================

        root_idx = 0

        clip = clip - clip[:, root_idx:root_idx + 1, :]

        # =========================================================
        # 11. CONVERT SKELETON TO TENSOR
        # =========================================================

        joints_tensor = torch.tensor(clip, dtype=torch.float32).permute(2, 0, 1)

        joints_tensor = joints_tensor.unsqueeze(-1)

        # =========================================================
        # 12. RETURN
        # =========================================================

        return (
            frames_tensor,    # [19, 16, 3, 224, 224]
            joints_tensor,    # [3, 64, 20, 1]
            label,
            camera_mask,      # [19]
            index,
        )
