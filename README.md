# Multimodal Transformer Fusion
This repository  introduces a transformer-based multimodal fusion architecture that integrates high-level visual and skeleton tokens extracted by  video and skeleton activity classification transformers.

# Multimodal Transformer Fusion of Skeleton and Multi-View Video Data for Arm-Action Recognition in Real-World Shop Environment

**Abstract:** Human arm-action recognition is a fundamental capability for automated retail environments and customer-aware robotic systems. 
In real-world stores, recognizing fine-grained customer actions such as picking up, holding, or wearing merchandise is challenging 
due to cluttered backgrounds, viewpoint variations, occlusions, and the subtle nature of arm movements. While RGB-based methods provide 
rich appearance and contextual information, skeleton-based approaches offer robust representations of human motion and body pose. 
In this paper, we investigate multimodal arm-action recognition by jointly exploiting synchronized visual and skeletal observations 
collected in a real retail hat shop. We introduce a new arm-action dataset containing synchronized RGB video frames and 3D skeleton 
sequences acquired from 115 customers in a real-world shopping environment. To effectively combine the complementary strengths of the 
two modalities, we propose a Transformer-based multimodal fusion architecture that integrates high-level visual and skeleton tokens extracted by 
video and skeleton activity classification transformers. The proposed model learns cross-modal interactions through self-attention 
and generates a unified representation for arm-action classification. Extensive experiments and ablation studies demonstrate that 
multimodal fusion consistently outperforms unimodal visual and skeleton-based baselines, highlighting the complementary nature of 
appearance and motion cues. The results show that the proposed approach provides an effective solution for robust arm-action recognition 
in realistic retail environments and offers a practical perception module for future automated retail stores and customer-aware service robots. 

**Our main contributions are as follows:**

--  We introduce a new dataset specifically designed for arm-action classification 
in a real-world retail environment. The dataset contains temporally synchronized skeleton data and RGB video captured simultaneously from multiple 
camera viewpoints, providing complementary information about human body configuration, motion, and appearance. The multi-camera setting is particularly 
important in realistic environments, where an individual may be partially or completely occluded in one camera view due to shelves, other customers, 
or changes in body orientation. The dataset therefore enables the investigation of multimodal action recognition under realistic conditions while 
explicitly considering the challenges associated with combining visual observations from different viewpoints.

-- Unlike existing approaches that typically rely on a single visual stream or combine 
skeleton data with visual information obtained from a single camera, we explicitly consider the case where visual observations of the same individual 
are available simultaneously from multiple cameras. We first aggregate the complementary visual representations obtained from different camera 
views and subsequently integrate the resulting visual representation with skeletal information. To this end, we propose a Transformer-based multimodal 
fusion architecture that learns interactions between skeleton and multi-camera visual features and produces a unified representation for arm-action classification. 
This formulation allows the model to exploit complementary information across viewpoints while reducing the impact of view-specific occlusions and missing visual cues.

-- We conduct extensive experiments to investigate the contribution 
of skeleton information, visual information, and multiple camera views to arm-action recognition. In addition to comparing unimodal and multimodal configurations, 
we evaluate different strategies for aggregating visual information from multiple cameras and analyze their effect on classification performance and training behavior. 
The results demonstrate the complementary nature of skeleton and visual modalities and show that incorporating synchronized observations from multiple camera 
viewpoints can provide additional information for robust arm-action classification in realistic retail environments.

<img width="1790" height="597" alt="Frame 1" src="https://github.com/user-attachments/assets/193a77ec-d83f-4b2c-82b1-a8406abab3b2" />

**Fig 1.** Illustration of the proposed multimodal fusion transformer. The proposed system takes synchronized video frames and skeleton sequences as input. These modalities 
				are processed by pre-trained transformer-based backbones to extract high-level token representations. The resulting visual and skeletal tokens are then 
				fused using the proposed multimodal fusion transformer, which learns complementary information across modalities. Finally, the fused representation is utilized to classify arm actions.

<img width="1441" height="604" alt="multimodal_visual_fusion_transformer" src="https://github.com/user-attachments/assets/79b6cc4a-6287-42fe-8b3b-0a6752e9e3de" />
**Fig 2.** Illustration of the proposed multi-camera visual token fusion transformer.

# 1. Data Collection 
The dataset was collected in a real retail hat shop located in Osaka, Japan. Data acquisition was conducted over 37 recording days between late 2024 and early 2025. 
To capture customer behavior throughout the store, a synchronized multi-view sensing system consisting of 19 Microsoft Azure Kinect DK RGB-D cameras was installed. 
The sensors were mounted on ceilings and shelves to provide overlapping coverage of customer activity areas. 
Layout of the hat shop from a bird’s-eye perspective, showing the deployment of 19 cameras and their coverage zones is given below. 

<img width="1132" height="713" alt="HatShop_Layout" src="https://github.com/user-attachments/assets/738590cc-4b52-4829-acdd-42a0f8f03fd2" />

**Fig 3.** Bird’s-eye view of the  hat shop showing the locations of the 19 cameras and their respective fields of view.

Each Azure Kinect sensor estimated a 32-joint three-dimensional skeleton for every detected customer. The skeletons from multiple cameras were transformed into a 
common world coordinate system and merged into continuous customer trajectories using an offline tracking framework. In addition to skeleton data, RGB videos were 
recorded and used for manual annotation of customer actions. The video recordings provided detailed visual information about customer interactions with merchandise, 
while the skeletons provided a privacy-preserving representation of body movements.

A total of 904 candidate customer visits were initially extracted from the recordings. After manual review, sessions involving groups, incomplete recordings, 
extremely short visits, and customers remaining outside the primary hat-shopping area were excluded. The final dataset contains 115 complete sessions of individual 
customers performing arm actions, each recorded continuously from store entry to exit.

Four arm-action categories were defined:

**Pick or Place Hat/Tie:** The customer reaches toward a shelf or counter to pick up or place a hat or a tie.

**Holding Hat/Tie:** The customer holds a hat or tie in one or both hands in front of the body.

**Wear or Take Off Hat:** The customer raises or lowers the arms to wear, remove, or adjust a hat on the head.

**Idle:** All remaining arm states, including standing with arms at the sides, hands in pockets, crossed arms, phone usage, or handling non-hat objects.

# 2. Requirements
## Environments
Following packages are required for this repo.

    - python 3.10.18+
    - torch  2.5+
    - torchvision 0.19+ 
    - CUDA 12.1+
    - cython 3.1.4+
    - scikit-learn 1.3+
    - numpy 2.2.6+
    - tqdm 4.67.3
    - matplotlib 3.7.5
    - opencv-contrib-python 5.0.0.93
    - pillow 12.2+
    - pytorch-cuda 12.4
    - pycocotools 2.0.11+
    - scipy 1.15.3+
    - scikit-learn 1.7.2+
    - scikit-image 0.25.2+
    - seaborn 0.13.2
    - torchvision 0.20.1+cu121
    - transformers 4.4+  

# 2. Training & Evaluation
## Training Multimodal Transformer Fusion Model
For training multimodal transformer fusion model using skeleton and multiple camera views, simply run  **'main_train_multimodal_fusion.py'**. This script initializes and trains a  
multimodal fusion model and a vision fusion model from scratch with randomly initialized weights. There are two options for vision fusion. Single-Query Camera Fusion is the best 
choice and it is the default mode. To choose attention based multi-camera fusion just activate the line 253  ''camera_token_fusion_model = CameraAttentionFusion().to(device)'' and
remove line 252.
## Training Multi-Camera Visual Token Fusion Models
For training multi-camera visual toke fusion models, simply run  **'main_train_visual_fusion.py'**. You can choose SingleQueryCameraFusion or CameraAttentionFusion models for training. 
Single-Query based fusion yields better accuracies and it converges much faster compared to  attention based multi-camera fusion model. VideoMAE V2 model is frozen and not updated, only 
fusion models are trained.
## Training Skeleton Activity Classification Model
Simply run **'main_train_skeleton.py'**. It fine-tunes from pre-trained Skate-former model. First, the backbone is frozen and only new added classification head is trained for 20 epochs. Then, 
entire network is trained for addition 80 epochs for a much smaller learning rate.
### Results
To reproduce the results given in the paper, run the script  **'main_test_multimodal_fusion.py'**.

# 3. Pre-trained Models and Data
## Data
Due to the privacy issues, we cannot distribute the video frames. The skeleton data used in this study are gievn under the data directory. They are written in python pickle files.
## Pre-trained Models
For pre-trained models, please contact us using the email address given below.

## Citation
```bibtex
@article{cevikalp2027,
  author    = {Hakan Cevikalp and Drazen Brscic and Zulkafil Abbas and Takayuki Kanda},
  title     = {Multimodal Transformer Fusion of Skeleton and Multi-View Video Data for Arm-Action Recognition in Real-World Shop Environment},
  journal = {Pattern Recognition},
  year      = {under review},
}

# Contact
If you have any question about our work, please do not hesitate to contact us by email hakan.cevikalp@gmail.com.


