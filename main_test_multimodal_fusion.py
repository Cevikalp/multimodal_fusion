import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from fusion_models import MultimodalFusionTransformer, VideoMAEActivityClassifierPretrained, SingleQueryCameraFusion, CameraAttentionFusion
from SkateFormer import SkateFormer_
from hatshop_dataset_multiple_cameras import HatShopSkateFormerDatasetPickleMultiple


# Data paths
skeleton_data_path = "/mnt/data2/skeletondata_test"
video_data_path = "/mnt/data2/visualdata_test_annotated"


# Output directory
output_dir = Path("arm_models/")
output_dir.mkdir(parents=True, exist_ok=True)

# Testing configuration
BS = 6
NUM_CLASSES = 4
NUM_TRIALS = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# pre-trained model directory
model_dir ='/home/robovie/cevikalpws/ActivityClassification/hakan/arm_models/checkpoint_model_fused_best.pth'


# ============================================================
# Create SkateFormer
# ============================================================

def create_model():

    # --------------------------------------------------------
    # Create the same architecture used during training
    # --------------------------------------------------------
    model_visual = VideoMAEActivityClassifierPretrained(
        checkpoint_path="/home/robovie/cevikalpws/VideoMAEv2/pretrained_models/vit_b_k710_dl_from_giant.pth",
        num_classes=4,
        num_frames=16,
        freeze_backbone=True,
    )

    model_visual = model_visual.to(device)
    model_visual.eval()

      # loading skeleton model
    model_joints = SkateFormer_(
        num_classes=10,
        num_people=1,
        num_points=20,
        num_frames=64,

        kernel_size=7,
        num_heads=32,

        attn_drop=0.5,
        head_drop=0.0,
        rel=True,
        drop_path=0.2,

        type_1_size=(8, 4),
        type_2_size=(8, 5),
        type_3_size=(8, 4),
        type_4_size=(8, 5),

        mlp_ratio=1.0,
        index_t=True,
    )
    # create the 4-class head
    in_features = model_joints.head.in_features
    model_joints.head = nn.Linear(in_features, 4)

    model_joints = model_joints.to(device)
    model_joints.eval()

    model_fusion = MultimodalFusionTransformer(num_classes=4)
    camera_token_fusion_model = SingleQueryCameraFusion()
   # camera_token_fusion_model = CameraAttentionFusion()

    checkpoint = torch.load(model_dir, map_location="cpu")

    model_fusion.load_state_dict(checkpoint["model_fusion_state_dict"])
    model_fusion = model_fusion.to(device)

    camera_token_fusion_model.load_state_dict(checkpoint["camera_token_fusion_model_state_dict"])
    camera_token_fusion_model=camera_token_fusion_model.to(device)
        
    camera_token_fusion_model.eval()
    model_fusion.eval()
    
       
        # The checkpoint was saved using:
    #
    # torch.save(model.state_dict(), ...)
    #
    # therefore it is directly a state_dict.
   

    return model_visual, model_joints, camera_token_fusion_model, model_fusion


# ============================================================
# Test one model
# ============================================================

def evaluate_model(model_visual, model_joints, model_fusion, camera_token_fusion_model, test_loader):

    all_preds = []
    all_labels = []

    # No gradients are required during testing
    with torch.no_grad():

        for frames, skeleton, labels, camera_mask, index_temp in test_loader:

            # ------------------------------------------------
            # Move skeleton and labels to GPU
            # ------------------------------------------------
           # skeleton = skeleton.to(device, non_blocking=True)
            frames = frames.to(device)
            camera_mask = camera_mask.to(device)
            labels = labels.to(device, non_blocking=True)
            skeleton = skeleton.to(device)
            bs = skeleton.size(0)
            index_t = torch.arange(64, device=device).unsqueeze(0).repeat(bs, 1)

            batch_size = skeleton.size(0)
            B, K, C, T, H, W = frames.shape

            # --------------------------------------------------
            # Which camera IDs are present in at least one sample
            # --------------------------------------------------

            valid_cameras = camera_mask.any(dim=0)
            camera_tokens = [None] * K
            token_shape = None

            for cam in range(K):
                # ----------------------------------------------
                # Skip cameras absent from the entire batch
                # ----------------------------------------------
                if not valid_cameras[cam]:
                    continue

                # ----------------------------------------------
                # [B,16,3,224,224]
                # ----------------------------------------------
                cam_frames = frames[:, cam]

                # ----------------------------------------------
                # VideoMAE
                # ----------------------------------------------
                tokens, logits = model_visual(cam_frames)

                # tokens:
                # [B,N,D]
                token_shape = tokens.shape

                # ----------------------------------------------
                # Zero-out samples where this camera is missing
                # ----------------------------------------------
                valid_samples = camera_mask[:, cam].float()
                tokens = tokens * valid_samples[:, None, None]
                camera_tokens[cam] = tokens

            # --------------------------------------------------
            # Fill cameras skipped above with zeros
            # --------------------------------------------------

            for cam in range(K):
                if camera_tokens[cam] is None:
                    camera_tokens[cam] = torch.zeros(token_shape, device=device, dtype=tokens.dtype)

            # --------------------------------------------------
            # [B,19,N,D]
            # --------------------------------------------------

            camera_tokens = torch.stack(camera_tokens, dim=1).to(device)
            feats_visual, logits  = camera_token_fusion_model(camera_tokens, camera_mask) 
            feats_joints, _ = model_joints(skeleton, index_t)
            feats_joints = feats_joints.permute(0, 2, 3, 1).flatten(1, 2)
            output = model_fusion(feats_visual, feats_joints)

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------
            logits = output['logits']
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    # Convert to NumPy arrays
    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------
    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    # --------------------------------------------------------
    # Macro F1
    # --------------------------------------------------------
    f1_macro = f1_score(
        all_labels,
        all_preds,
        average="macro",
        labels=np.arange(NUM_CLASSES),
        zero_division=0
    )

    # --------------------------------------------------------
    # Weighted F1
    # --------------------------------------------------------
    f1_weighted = f1_score(
        all_labels,
        all_preds,
        average="weighted",
        labels=np.arange(NUM_CLASSES),
        zero_division=0
    )

    # --------------------------------------------------------
    # F1 for each individual class
    # --------------------------------------------------------
    f1_per_class = f1_score(
        all_labels,
        all_preds,
        average=None,
        labels=np.arange(NUM_CLASSES),
        zero_division=0
    )

    return (
        accuracy,
        f1_macro,
        f1_weighted,
        f1_per_class
    )


# ============================================================
# Main testing function
# ============================================================

def test():

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    dataset_test = HatShopSkateFormerDatasetPickleMultiple(
        video_path=video_data_path,
        skeleton_path=skeleton_data_path,
        clip_size=16
    )

    print(
        f"\nTest dataset size: {len(dataset_test)}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------
    test_loader = DataLoader(
        dataset_test,
        batch_size=BS,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Create and load model
    # --------------------------------------------------------
    model_visual, model_joints, camera_token_fusion_model, model_fusion = create_model()

   
    print("\nModel loaded successfully.")
    print("Model is in evaluation mode.")

    # ========================================================
    # Storage for 10 trials
    # ========================================================

    Accuracy_all = np.zeros(NUM_TRIALS)
    F1_macro_all = np.zeros(NUM_TRIALS)
    F1_weighted_all = np.zeros(NUM_TRIALS)

    # Shape:
    # [trial, class]
    #
    # NOT [class, trial]
    #
    # This makes the statistics much easier to calculate.
    F1_per_class_all = np.zeros(
        (NUM_TRIALS, NUM_CLASSES)
    )

    # ========================================================
    # Run trials
    # ========================================================

    for trial in range(NUM_TRIALS):

        print(
            f"\n========================================"
        )
        print(
            f"Trial {trial + 1}/{NUM_TRIALS}"
        )
        print(
            f"========================================"
        )

        (accuracy, f1_macro, f1_weighted, f1_per_class ) = evaluate_model(model_visual, model_joints, model_fusion, camera_token_fusion_model, test_loader)

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------
        Accuracy_all[trial] = accuracy

        F1_macro_all[trial] = f1_macro

        F1_weighted_all[trial] = f1_weighted

        F1_per_class_all[trial, :] = f1_per_class

        # ----------------------------------------------------
        # Print results for this trial
        # ----------------------------------------------------
        print(
            f"Accuracy      : {accuracy:.4f}"
        )

        print(
            f"Macro F1      : {f1_macro:.4f}"
        )

        print(
            f"Weighted F1   : {f1_weighted:.4f}"
        )

        for class_idx, f1 in enumerate(f1_per_class):
            print(
                f"Class {class_idx} F1    : {f1:.4f}"
            )

    # ========================================================
    # Calculate mean and standard deviation
    # ========================================================

    mean_accuracy = np.mean(
        Accuracy_all
    )

    std_accuracy = np.std(
        Accuracy_all,
        ddof=1
    )

    mean_f1_macro = np.mean(
        F1_macro_all
    )

    std_f1_macro = np.std(
        F1_macro_all,
        ddof=1
    )

    mean_f1_weighted = np.mean(
        F1_weighted_all
    )

    std_f1_weighted = np.std(
        F1_weighted_all,
        ddof=1
    )

    # --------------------------------------------------------
    # Per-class mean and std
    # --------------------------------------------------------
    mean_f1_classes = np.mean(
        F1_per_class_all,
        axis=0
    )

    std_f1_classes = np.std(
        F1_per_class_all,
        axis=0,
        ddof=1
    )

    # ========================================================
    # Print final results
    # ========================================================

    print("\n\n")
    print("======================================================")
    print("FINAL RESULTS")
    print("======================================================")

    print(
        f"Accuracy       : "
        f"{mean_accuracy:.4f} ± {std_accuracy:.4f}"
    )

    print(
        f"Macro F1       : "
        f"{mean_f1_macro:.4f} ± {std_f1_macro:.4f}"
    )

    print(
        f"Weighted F1    : "
        f"{mean_f1_weighted:.4f} ± {std_f1_weighted:.4f}"
    )

    print("\nPer-class F1:")

    for class_idx in range(NUM_CLASSES):

        print(
            f"Class {class_idx} F1 : "
            f"{mean_f1_classes[class_idx]:.4f} "
            f"± {std_f1_classes[class_idx]:.4f}"
        )

    print("======================================================")

    # --------------------------------------------------------
    # Also print individual trial results
    # --------------------------------------------------------

    print("\nIndividual trial results:")

    for trial in range(NUM_TRIALS):

        print(
            f"Trial {trial + 1:2d}: "
            f"Accuracy={Accuracy_all[trial]:.4f}, "
            f"Macro F1={F1_macro_all[trial]:.4f}, "
            f"Weighted F1={F1_weighted_all[trial]:.4f}"
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    test()