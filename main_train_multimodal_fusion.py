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
from fusion_models import  MultimodalFusionTransformer, VideoMAEActivityClassifierPretrained, SingleQueryCameraFusion, CameraAttentionFusion, FocalLoss
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import numpy as np

   
# Data Paths
# train
skeleton_data_path = '/mnt/data2/skeletondata_new'
video_data_path = '/mnt/data2/visualdata_annotated'
#test
skeleton_test_data_path = '/mnt/data2/skeletondata_test'
video_test_data_path = '/mnt/data2/visualdata_test_annotated'


output_dir = Path("arm_models/")
output_dir.mkdir(exist_ok=True)

BS = 8
EPOCHS = 100
NUM_CLASSES = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def validate_test(
    model_visual,
    camera_token_fusion_model,
    model_joints,
    model_fusion,
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

    model_visual.eval()
    model_joints.eval()
    model_fusion.eval()

    all_preds = []
    all_labels = []

    for frames, skeleton, labels, camera_mask, index_temp in test_loader:

        skeleton = skeleton.to(device)
        frames = frames.to(device)
        camera_mask = camera_mask.to(device)

        bs = skeleton.size(0)
        index_t = torch.arange(64, device=device).unsqueeze(0).repeat(bs, 1)

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

        camera_tokens = torch.stack(camera_tokens, dim=1)
        feats_visual, _ = camera_token_fusion_model(camera_tokens, camera_mask)


        #feats_visual, _ = model_visual(frames)
        feats_joints, _ = model_joints(skeleton, index_t)
        feats_joints = feats_joints.permute(0, 2, 3, 1).flatten(1, 2)

        output = model_fusion(feats_visual, feats_joints)
        logits = output['logits']

      

        # logits shape: [B, 4]
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

    return ( accuracy, f1_macro, f1_weighted, f1_per_class )
    


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
     
    # Initially freeze the model excpet for the classification head
    for param in model_joints.parameters():
        param.requires_grad = False

    for param in model_joints.head.parameters():
        param.requires_grad = True


   # my_checkpoint_path = ""
   # checkpoint = torch.load(my_checkpoint_path, map_location=device)
   # model_joints.load_state_dict(checkpoint['model_state_dict'])
    model_joints = model_joints.to(device)
   

    # loading MAE V2 visual model
    model_visual = VideoMAEActivityClassifierPretrained(
        checkpoint_path="/home/robovie/cevikalpws/VideoMAEv2/pretrained_models/vit_b_k710_dl_from_giant.pth",
        num_classes=4,
        num_frames=16,
        freeze_backbone=True
    )

    model_visual = model_visual.to(device)
    camera_token_fusion_model = SingleQueryCameraFusion().to(device)
   # camera_token_fusion_model = CameraAttentionFusion().to(device)

    # fusion Model
    model_fusion = MultimodalFusionTransformer(num_classes=4)
  #  my_checkpoint_path ="/home/robovie/cevikalpws/ActivityClassification/hakan2/arm_models/model_epoch_fused_transformer_best.pth"

   # state_dict = torch.load(my_checkpoint_path, map_location=device)
    #model_fusion.load_state_dict(state_dict)
    model_fusion =  model_fusion.to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    #criterion = VideoClassificationLoss()
    optimizer = torch.optim.Adam(model_fusion.parameters(), lr=1e-4)
    optimizer_visual = torch.optim.Adam(camera_token_fusion_model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scheduler_visual = CosineAnnealingLR(optimizer_visual, T_max=EPOCHS, eta_min=1e-6)

    epochs = EPOCHS
    best_accuracy = 0.0

    for epoch in range(epochs):

        camera_token_fusion_model.train()
        model_fusion.train()
  
        running_loss = 0.0

        for frames, skeleton, labels, camera_mask, index_temp in train_loader:

             # [T,C,H,W] -> [1,C,T,H,W]
              #frames = frames.unsqueeze(0)
            frames = frames.to(device)
            camera_mask = camera_mask.to(device)

            skeleton = skeleton.to(device)
            batch_size = skeleton.size(0)
            index_t = (torch.arange(64).unsqueeze(0).repeat(batch_size, 1))

            index_t = index_t.to(device)

            labels = labels.to(device)

            # Fusion of visual camera tokens
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
            feats_visual, logits_temp  = camera_token_fusion_model(camera_tokens, camera_mask) 

            optimizer.zero_grad()
            optimizer_visual.zero_grad()

           # feats_visual, _ = model_visual(frames)
            feats_joints, _ = model_joints(skeleton, index_t)
            feats_joints = feats_joints.permute(0, 2, 3, 1).flatten(1, 2)

            output = model_fusion(feats_visual, feats_joints)
            #logits = output['logits'].sigmoid()
            logits = output['logits']

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer_visual.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {avg_loss:.4f}"
        )

        if scheduler:
            scheduler.step()
            scheduler_visual.step()


         
        if True:
            Accuracy_all = np.zeros(6)
            F1_macro_all = np.zeros(6)
            F1_weighted_all = np.zeros(6)

            # Shape:
            # [trial, class]
            #
            # NOT [class, trial]
            #
            # This makes the statistics much easier to calculate.
            F1_per_class_all = np.zeros((6,4))
            model_visual,
    
            for trial in range(2):
                (accuracy, f1_macro, f1_weighted, f1_per_class ) = validate_test(model_visual, camera_token_fusion_model, model_joints, model_fusion, dataset_test)
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
                print(f"Accuracy      : {accuracy:.4f}")
        
                print(f"Macro F1      : {f1_macro:.4f}")
        
                print(f"Weighted F1   : {f1_weighted:.4f}")
        
                for class_idx, f1 in enumerate(f1_per_class):
                    print(
                        f"Class {class_idx} F1    : {f1:.4f}"
                    )
                
            # ========================================================
            # Calculate mean and standard deviation
            # ========================================================
        
            mean_accuracy = np.mean(Accuracy_all)
        
            std_accuracy = np.std(Accuracy_all, ddof=1)
        
            mean_f1_macro = np.mean(F1_macro_all)

            test_accuracy =  (mean_accuracy + mean_f1_macro)/2.0
        
            std_f1_macro = np.std(F1_macro_all, ddof=1)
        
            mean_f1_weighted = np.mean(F1_weighted_all)
        
            std_f1_weighted = np.std(F1_weighted_all, ddof=1)
        
            # --------------------------------------------------------
            # Per-class mean and std
            # --------------------------------------------------------
            mean_f1_classes = np.mean(F1_per_class_all, axis=0)
        
            std_f1_classes = np.std(F1_per_class_all, axis=0, ddof=1)
        
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
            
        if True:
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_fusion_state_dict': model_fusion.state_dict(),
                    'camera_token_fusion_model_state_dict':camera_token_fusion_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    }, output_dir / "checkpoint_model_fused_best_6_layers.pth")

            torch.save({
                'epoch': epoch,
                'model_fusion_state_dict': model_fusion.state_dict(),
                'camera_token_fusion_model_state_dict':camera_token_fusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                }, output_dir / "checkpoint_model_fused_6_layers.pth")
                

if __name__ == "__main__":
    train()