# multimodal_fusion
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

<img width="2164" height="713" alt="fig1" src="https://github.com/user-attachments/assets/baf2d2e0-22c9-4ad4-81a9-c748e463872f" />

**Fig 1.** Illustration of the proposed multimodal fusion transformer. The proposed system takes synchronized video frames and skeleton sequences as input. These modalities 
				are processed by pre-trained transformer-based backbones to extract high-level token representations. The resulting visual and skeletal tokens are then 
				fused using the proposed multimodal fusion transformer, which learns complementary information across modalities. Finally, the fused representation is utilized to classify arm actions.


