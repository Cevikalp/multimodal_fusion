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
from hatshop_skateformer_dataset import HatShopSkateFormerDatasetPickle
#from models_pretrained_3 import VideoMAEActivityClassifierPretrained, VideoClassificationLoss
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets, reduction='none'
        )

        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    
# Data Paths
skeleton_data_path = '/mnt/data2/skeletondata_new'
video_data_path = '/mnt/data2/visualdata_annotated'

output_dir = Path("arm_models/")
output_dir.mkdir(exist_ok=True)

BS = 24 
EPOCHS = 150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



@torch.no_grad()
def validate_test(
    model,
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

    all_preds = []
    all_labels = []

    for frames, skeleton, labels, index_temp in test_loader:

        skeleton = skeleton.to(device)
        frames = frames.to(device)
        
        skeleton = skeleton.to(device)
        batch_size = skeleton.size(0)
        index_t = (torch.arange(64).unsqueeze(0).repeat(batch_size, 1))

        index_t = index_t.to(device)

        labels = labels.to(device)
        feats, logits = model(skeleton,index_t)
        
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
 
    dataset_train = HatShopSkateFormerDatasetPickle(
        video_path=video_data_path,
        skeleton_path=skeleton_data_path,      # or "arm"
        clip_size=32
    )

    print(f"Dataset size: {len(dataset_train)}")

    train_loader = DataLoader(
        dataset_train,
        batch_size=BS,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    dataset_test = HatShopSkateFormerDatasetPickle(
        video_path=video_data_path,
        skeleton_path=skeleton_data_path,      # or "arm"
        clip_size=32
    )

    print(f"Dataset size: {len(dataset_train)}")

    # --------------------------------------------------
    # create model EXACTLY as trained
    # --------------------------------------------------
     # Pretrained checkpoint path
    weight_path = "/home/robovie/cevikalpws/ActivityClassification/skeleton_model/pretrained/NWUCLA/SkateFormer_b.pt"
    
    model = SkateFormer_(
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

    print("Loading checkpoint...")
    checkpoint = torch.load(weight_path, map_location="cpu")


    missing, unexpected = model.load_state_dict(
        checkpoint,
        strict=False
    )

    # Setting the class number to 4 for the new dataset
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, 4)

    # Initially freeze the model excpet for the classification head
    for param in model.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = True

    model.to(device)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))
   
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    #criterion = VideoClassificationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    epochs = EPOCHS
    best_accuracy = 0.0

    for epoch in range(epochs):

        if epoch == 40:
            print("Unfreezing entire model...")

            for param in model.parameters():
                param.requires_grad = True

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=1e-5
            )

            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=epochs - 10,
                eta_min=1e-6
            )


        if True:

            test_results = validate_test(model,dataset_test)
            print(f"Accuracy      : {test_results['accuracy']:.4f}")
            print(f"Macro F1      : {test_results['f1_macro']:.4f}")
            print(f"Weighted F1   : {test_results['f1_weighted']:.4f}")
            test_accuracy = test_results['accuracy']
        
            for i, f1 in enumerate(test_results['f1_per_class']):
                print(f"Class {i} F1    : {f1:.4f}")

  
        running_loss = 0.0

        for frames, skeleton, labels, index_temp in train_loader:

             # [T,C,H,W] -> [1,C,T,H,W]
            

            #frames = frames.unsqueeze(0)
            frames = frames.to(device)

            skeleton = skeleton.to(device)
            batch_size = skeleton.size(0)
            index_t = (torch.arange(64).unsqueeze(0).repeat(batch_size, 1))

            index_t = index_t.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()
            feats, logits = model(skeleton, index_t)
           # logits = logits.sigmoid()
                    
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
                torch.save(model.state_dict(), output_dir / f"model_epoch_skeleton_transformer_normal_best.pth")
            
            torch.save(model.state_dict(), output_dir / f"model_epoch_skeleton_transformer_normal.pth")


if __name__ == "__main__":
    train()