from xml.parsers.expat import model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import glob
import h5py
from SkateFormer import SkateFormer_
from hatshop_dataset_multiple_cameras import HatShopSkateFormerDatasetPickleMultiple
from fusion_models import VideoMAEActivityClassifierPretrained, CameraAttentionFusion, FocalLoss
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path


# train data Paths
skeleton_data_path = '/mnt/data2/skeletondata_new'
video_data_path = '/mnt/data2/visualdata_annotated'
#test
skeleton_test_data_path = '/mnt/data2/skeletondata_test'
video_test_data_path = '/mnt/data2/visualdata_test_annotated'


output_dir = Path("arm_models/")
output_dir.mkdir(exist_ok=True)

BS = 24 
EPOCHS = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def validate_test(
    model,
    camera_token_fusion_model,
    dataset_test,
    device=device,
    batch_size=4
):

    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    model.eval()
    camera_token_fusion_model.eval()

    all_preds = []
    all_labels = []

    for frames, skeleton, labels, camera_mask, index_temp in test_loader:



        frames = frames.to(device)
        camera_mask = camera_mask.to(device)

        skeleton = skeleton.to(device)

        batch_size = skeleton.size(0)

        index_t = (torch.arange(64).unsqueeze(0).repeat(batch_size, 1)).to(device)

        labels = labels.to(device)

        # --------------------------------------------------
        # frames:
        # [B, 19, 16, 3, 224, 224]
        # --------------------------------------------------

        B, K, T, C, H, W = frames.shape

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
            tokens, logits = model(cam_frames)

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

        camera_tokens = torch.stack(camera_tokens, dim=1)
        fused_tokens, logits, weights = camera_token_fusion_model(camera_tokens, camera_mask)     

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    f1_macro = f1_score(
        all_labels,
        all_preds,
        average='macro'
    )

    f1_weighted = f1_score(
        all_labels,
        all_preds,
        average='weighted'
    )

    f1_per_class = f1_score(
        all_labels,
        all_preds,
        average=None
    )

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_per_class": f1_per_class
    }
    


##############################################
# Training
##############################################

def train():
 
    dataset_train = HatShopSkateFormerDatasetPickleMultiple(
        video_path=video_data_path,
        skeleton_path=skeleton_data_path,      # or "arm"
        clip_size=16
    )

    print(f"Dataset size: {len(dataset_train)}")

    train_loader = DataLoader(
        dataset_train,
        batch_size=BS,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    dataset_test = HatShopSkateFormerDatasetPickleMultiple(
        video_path=video_test_data_path,
        skeleton_path=skeleton_test_data_path,      # or "arm"
        clip_size=16
    )

    print(f"Dataset size: {len(dataset_test)}")



    model = VideoMAEActivityClassifierPretrained(
        checkpoint_path="/home/robovie/cevikalpws/VideoMAEv2/pretrained_models/vit_b_k710_dl_from_giant.pth",
        num_classes=4,
        num_frames=16,
        freeze_backbone=True,
    )
    
    model = model.to(device)
    model.eval()
    
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    #criterion = VideoClassificationLoss()
  
    camera_token_fusion_model = CameraAttentionFusion().to(device)
    #weight_path = "/home/robovie/cevikalpws/ActivityClassification/hakan3/arm_models/model_epoch_visual_transformer.pth"
    #checkpoint = torch.load(weight_path, map_location="cpu")

    # The checkpoint was saved using:
    #
    # torch.save(model.state_dict(), ...)
    #
    # therefore it is directly a state_dict.
    #camera_token_fusion_model.load_state_dict(checkpoint, strict=True)
    #camera_token_fusion_model = camera_token_fusion_model.to(device)
    optimizer = torch.optim.Adam(camera_token_fusion_model.parameters(), lr=0.51e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    epochs = EPOCHS
    best_accuracy = 0.0

    for epoch in range(epochs):

       
        if True:

            test_results = validate_test(model,camera_token_fusion_model,dataset_test)
            print(f"Accuracy      : {test_results['accuracy']:.4f}")
            print(f"Macro F1      : {test_results['f1_macro']:.4f}")
            print(f"Weighted F1   : {test_results['f1_weighted']:.4f}")
            test_accuracy = test_results['accuracy']
        
            for i, f1 in enumerate(test_results['f1_per_class']):
                print(f"Class {i} F1    : {f1:.4f}")

  
        running_loss = 0.0

        for frames, skeleton, labels, camera_mask, index_temp in train_loader:

             # [T,C,H,W] -> [1,C,T,H,W]

            frames = frames.to(device)
            camera_mask = camera_mask.to(device)

            skeleton = skeleton.to(device)

            batch_size = skeleton.size(0)

            index_t = (torch.arange(64).unsqueeze(0).repeat(batch_size, 1)).to(device)

            labels = labels.to(device)

            # --------------------------------------------------
            # frames:
            # [B, 19, 16, 3, 224, 224]
            # --------------------------------------------------

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
                tokens, logits = model(cam_frames)

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
            fused_tokens, logits, weights = camera_token_fusion_model(camera_tokens, camera_mask)     
               # camera_tokens:
            # [B,19,N,D]
           
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {avg_loss:.4f}"
        )

        if scheduler:
            scheduler.step()

        if True:

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(camera_token_fusion_model.state_dict(), output_dir / f"model_epoch_visual_transformer_best.pth")
            
            torch.save(camera_token_fusion_model.state_dict(), output_dir / f"model_epoch_visual_transformer.pth")


if __name__ == "__main__":
    train()