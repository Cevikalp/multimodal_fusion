import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling_finetune import vit_base_patch16_224
import math


class MultimodalFusionTransformer(nn.Module):
    """
    Fuse VideoMAE and SkateFormer tokens using
    a learnable fusion CLS token.

    Inputs
    ------
    video_tokens : [B, Nv, Dv]
    skel_tokens  : [B, Ns, Ds]

    Output
    ------
    logits : [B, num_classes]
    """

    def __init__(
        self,
        video_dim=768,
        skel_dim=192,
        fusion_dim=256,
        num_classes=4,
        num_layers=5,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        # ----------------------------------
        # Project both modalities
        # to common dimension
        # ----------------------------------

        self.video_proj = nn.Linear(
            video_dim,
            fusion_dim
        )

        self.skel_proj = nn.Linear(
            skel_dim,
            fusion_dim
        )

        # ----------------------------------
        # Fusion CLS token
        # ----------------------------------

        self.fusion_cls = nn.Parameter(
            torch.randn(
                1,
                1,
                fusion_dim
            )
        )

        # ----------------------------------
        # Modality embeddings
        # ----------------------------------

        self.video_type_embed = nn.Parameter(
            torch.randn(
                1,
                1,
                fusion_dim
            )
        )

        self.skel_type_embed = nn.Parameter(
            torch.randn(
                1,
                1,
                fusion_dim
            )
        )

        self.cls_type_embed = nn.Parameter(
            torch.randn(
                1,
                1,
                fusion_dim
            )
        )

        # ----------------------------------
        # Fusion transformer
        # ----------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=int(
                fusion_dim * mlp_ratio
            ),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(
            fusion_dim
        )

        # ----------------------------------
        # Classification head
        # ----------------------------------

        self.head = nn.Sequential(
            nn.Linear(
                fusion_dim,
                fusion_dim
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                fusion_dim,
                num_classes
            )
        )

    def forward(
        self,
        video_tokens,
        skel_tokens,
    ):
        """
        video_tokens:
            [B, Nv, Dv]

        skel_tokens:
            [B, Ns, Ds]
        """

        B = video_tokens.shape[0]

        # ----------------------------------
        # Project to common space
        # ----------------------------------

        video_tokens = self.video_proj(
            video_tokens
        )

        skel_tokens = self.skel_proj(
            skel_tokens
        )

        # ----------------------------------
        # Add modality embeddings
        # ----------------------------------

        video_tokens = (
            video_tokens
            + self.video_type_embed
        )

        skel_tokens = (
            skel_tokens
            + self.skel_type_embed
        )

        cls_token = (
            self.fusion_cls.expand(
                B,
                -1,
                -1
            )
            + self.cls_type_embed
        )

        # ----------------------------------
        # Concatenate
        # ----------------------------------

        tokens = torch.cat(
            [
                cls_token,
                video_tokens,
                skel_tokens,
            ],
            dim=1
        )

        # ----------------------------------
        # Fusion transformer
        # ----------------------------------

        tokens = self.fusion_transformer(
            tokens
        )

        # ----------------------------------
        # CLS feature
        # ----------------------------------

        fusion_feat = self.norm(
            tokens[:, 0]
        )

        logits = self.head(
            fusion_feat
        )

        return {
            "logits": logits,
            "fusion_feat": fusion_feat,
        }

class SingleQueryCameraFusion(nn.Module):
    def __init__(
        self,
        token_dim=768,
        hidden_dim=256,
        max_cameras=19,
        num_classes=4,
        num_heads=8,
        num_layers=3,
        dropout=0.1
    ):
        super().__init__()

        self.token_dim = token_dim

        # ==================================================
        # SINGLE Learnable Query (Content + Positional)
        # ==================================================
        self.query_content = nn.Parameter(torch.empty(1, 1, token_dim))
        self.query_pos = nn.Parameter(torch.empty(1, 1, token_dim))

        self.camera_embeddings = nn.Parameter(torch.empty(max_cameras, token_dim))

        # Gate normalization to keep multiplicative weights stable
        self.gate_norm = nn.LayerNorm(token_dim)

        # Standard Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.query_content, std=0.02)
        nn.init.trunc_normal_(self.query_pos, std=0.02)
        nn.init.trunc_normal_(self.camera_embeddings, std=0.02)

    def forward(self, camera_tokens, camera_mask):
        """
        camera_tokens : [B, K, N, D]
        camera_mask   : [B, K] (1 = valid, 0 = missing; guaranteed >= 1 active)
        """
        B, K, N, D = camera_tokens.shape

        # 1. Add Camera Embeddings
        cam_ids = torch.arange(K, device=camera_tokens.device)
        cam_embed = self.camera_embeddings[cam_ids].view(1, K, 1, D)
        embedded_tokens = camera_tokens + cam_embed

        # 2. Flatten for Cross-Attention -> [B, K*N, D]
        memory = embedded_tokens.reshape(B, K * N, D)

        # 3. Create Key Padding Mask
        token_mask = camera_mask.unsqueeze(-1).expand(-1, -1, N).reshape(B, K * N)
        memory_key_padding_mask = (~token_mask.bool())

        # 4. Run Decoder with Single Query [B, 1, D]
        single_query = (self.query_content + self.query_pos).expand(B, -1, -1)
        fused_query = self.decoder(
            tgt=single_query,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask
        ) # Output shape: [B, 1, D]

        # =========================================================================
        # 5. Product: camera_tokens * fused_query
        # =========================================================================
        gate = self.gate_norm(fused_query).unsqueeze(1) # [B, 1, 1, D]
        
        # Elementwise broadcast product: [B, K, N, D] * [B, 1, 1, D]
        gated_camera_tokens = camera_tokens * gate

        # 6. Mask Missing Cameras and Pool K -> Output [B, N, D]
        valid_cam_mask = camera_mask.unsqueeze(-1).unsqueeze(-1) # [B, K, 1, 1]
        gated_camera_tokens = gated_camera_tokens * valid_cam_mask

        # Average across active cameras
        active_cam_count = camera_mask.sum(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1) # [B, 1, 1, 1]
        fused_tokens = gated_camera_tokens.sum(dim=1) / active_cam_count.squeeze(1) # [B, N, D]

        # 7. Classification Head
        logits = self.classifier(fused_tokens.mean(dim=1))

        return fused_tokens, logits

class DETRQueryResampledCameraFusion(nn.Module):
    def __init__(
        self,
        token_dim=768,
        hidden_dim=256,
        num_queries=64,
        max_cameras=19,
        num_classes=4,
        num_heads=8,
        num_layers=3,
        dropout=0.1
    ):
        super().__init__()

        self.token_dim = token_dim
        self.num_queries = num_queries

        # DETR Content & Positional Queries [1, 256, D]
        self.query_content = nn.Parameter(torch.empty(1, num_queries, token_dim))
        self.query_pos = nn.Parameter(torch.empty(1, num_queries, token_dim))

        self.camera_embeddings = nn.Parameter(torch.empty(max_cameras, token_dim))

        # Standard Transformer Decoder Stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Map 256 query slots down to a dynamic gate matching memory dimensions
        self.query_to_spatial = nn.Sequential(
            nn.Linear(num_queries, 1),
            nn.Sigmoid()  # Restricts multiplicative gate between 0.0 and 1.0
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.query_content, std=0.02)
        nn.init.trunc_normal_(self.query_pos, std=0.02)
        nn.init.trunc_normal_(self.camera_embeddings, std=0.02)

    def forward(self, camera_tokens, camera_mask):
        """
        camera_tokens : [B, K, N, D]
        camera_mask   : [B, K] (1 = valid, 0 = missing; guaranteed >= 1 active)
        """
        B, K, N, D = camera_tokens.shape

        # 1. Camera Embeddings
        cam_ids = torch.arange(K, device=camera_tokens.device)
        cam_embed = self.camera_embeddings[cam_ids].view(1, K, 1, D)
        embedded_tokens = camera_tokens + cam_embed

        # 2. Flatten for Cross-Attention
        memory = embedded_tokens.reshape(B, K * N, D)

        # 3. Mask missing camera tokens
        token_mask = camera_mask.unsqueeze(-1).expand(-1, -1, N).reshape(B, K * N)
        memory_key_padding_mask = (~token_mask.bool())

        # 4. Run Transformer Decoder -> Output: [B, 64, D]
        queries = (self.query_content + self.query_pos).expand(B, -1, -1)
        fused_queries = self.decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # =========================================================================
        # 5. Product (memory * fused_queries) & Collapse to [B, N, D]
        # =========================================================================
        
        # Project fused queries from [B, 64, D] -> [B, D, 64] -> [B, D, 1] -> [B, 1, 1, D]
        # Generates global spatial modulation weights
        query_gate = self.query_to_spatial(fused_queries.transpose(1, 2)).transpose(1, 2) # [B, 1, D]
        query_gate = query_gate.unsqueeze(1) # [B, 1, 1, D]

        # Multiplicative interaction: camera_tokens * fused_query_gate
        # Shape: [B, K, N, D] * [B, 1, 1, D] -> [B, K, N, D]
        gated_memory = camera_tokens * query_gate

        # 6. Apply Camera Mask & Pool across active cameras K -> [B, N, D]
        cam_mask_expanded = camera_mask.unsqueeze(-1).unsqueeze(-1) # [B, K, 1, 1]
        gated_memory = gated_memory * cam_mask_expanded

        # Divide by active camera counts to maintain exact scale
        active_cams = camera_mask.sum(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1) # [B, 1, 1, 1]
        fused_tokens = gated_memory.sum(dim=1) / active_cams.squeeze(1) # [B, N, D]

        # =========================================================================
        # 7. Classification
        # =========================================================================
        fused_global = fused_tokens.mean(dim=1)
        logits = self.classifier(fused_global)

        return fused_tokens, logits


class QueryCameraFusion(nn.Module):
    """
    Variable-camera token fusion using learnable queries.

    Input
    -----
    camera_tokens : [B, K, N, D]
        K = max number of cameras
        N = tokens per camera

    camera_mask : [B, K]
        1 = camera exists
        0 = missing camera

    Output
    ------
    fused_tokens : [B, num_queries, D]
    """

    def __init__(
        self,
        token_dim=768,
        num_queries=512,
        max_cameras=6,
        num_heads=8,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.num_queries = num_queries
        self.max_cameras = max_cameras

        # --------------------------------------------------
        # Learnable output queries
        # --------------------------------------------------
        self.queries = nn.Parameter(
            torch.randn(1, num_queries, token_dim)
        )

        # --------------------------------------------------
        # Camera ID embeddings
        # Camera-0, Camera-1, ...
        # --------------------------------------------------
        self.camera_embeddings = nn.Parameter(
            torch.randn(max_cameras, token_dim)
        )

        # --------------------------------------------------
        # Presence embeddings
        # [0] = missing camera
        # [1] = valid camera
        # --------------------------------------------------
        self.presence_embeddings = nn.Parameter(
            torch.randn(2, token_dim)
        )

        self.layers = nn.ModuleList()

        for _ in range(num_layers):

            self.layers.append(
                nn.ModuleDict(
                    {
                        "cross_attn": nn.MultiheadAttention(
                            embed_dim=token_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),

                        "norm1": nn.LayerNorm(token_dim),

                        "ffn": nn.Sequential(
                            nn.Linear(token_dim, token_dim * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(token_dim * 4, token_dim),
                        ),

                        "norm2": nn.LayerNorm(token_dim),
                    }
                )
            )

    def forward(
        self,
        camera_tokens,
        camera_mask=None,
    ):
        """
        camera_tokens : [B,K,N,D]
        camera_mask   : [B,K]
        """

        B, K, N, D = camera_tokens.shape

        assert K <= self.max_cameras

        # ==================================================
        # Camera ID embeddings
        # ==================================================

        cam_ids = torch.arange(
            K,
            device=camera_tokens.device
        )

        cam_embed = self.camera_embeddings[cam_ids]
        cam_embed = cam_embed.view(
            1, K, 1, D
        )

        camera_tokens = camera_tokens + cam_embed

        # ==================================================
        # Presence embeddings
        # ==================================================

        if camera_mask is not None:

            presence_embed = self.presence_embeddings[
                camera_mask.long()
            ]

            presence_embed = presence_embed.unsqueeze(2)

            camera_tokens = (
                camera_tokens + presence_embed
            )

        # ==================================================
        # Flatten cameras
        # ==================================================

        memory = camera_tokens.reshape(
            B,
            K * N,
            D,
        )

        # ==================================================
        # Build token mask
        # ==================================================

        key_padding_mask = None

        if camera_mask is not None:

            token_mask = (
                camera_mask
                .unsqueeze(-1)
                .expand(-1, -1, N)
                .reshape(B, K * N)
            )

            key_padding_mask = ~token_mask.bool()

        # ==================================================
        # Queries
        # ==================================================

        q = self.queries.expand(
            B,
            -1,
            -1,
        )

        # ==================================================
        # Query cross-attention
        # ==================================================

        for layer in self.layers:

            attn_out, _ = layer["cross_attn"](
                query=q,
                key=memory,
                value=memory,
                key_padding_mask=key_padding_mask,
            )

            q = layer["norm1"](
                q + attn_out
            )

            ffn_out = layer["ffn"](q)

            q = layer["norm2"](
                q + ffn_out
            )

        return q

class CameraAttentionFusion(nn.Module):
    """
    Multi-camera attention fusion followed by classification.

    Input:
        camera_tokens : [B, K, N, D]
            B = batch size
            K = number of cameras (19)
            N = number of tokens per camera
            D = token dimension (e.g. 768)

        camera_mask : [B, K]
            1 = camera available
            0 = camera missing

    Output:
        logits        : [B, num_classes]
        weights       : [B, K]
            Attention weight assigned to each camera
    """

    def __init__(
        self,
        token_dim=768,
        num_classes=4,
        hidden_dim=256,
        dropout=0.1
    ):
        super().__init__()

        # -------------------------------------------------
        # Camera attention / scoring network
        # -------------------------------------------------
        self.score_net = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, 1)
        )

        # -------------------------------------------------
        # Classification head
        # -------------------------------------------------
        self.classifier = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, camera_tokens, camera_mask=None):

        # camera_tokens: [B, K, N, D]
        B, K, N, D = camera_tokens.shape

        # =================================================
        # 1. Compute one descriptor for each camera
        # =================================================
        # [B, K, N, D] -> [B, K, D]
        camera_descriptors = camera_tokens.mean(dim=2)

        # =================================================
        # 2. Compute camera scores
        # =================================================
        # [B, K, D] -> [B, K, 1] -> [B, K]
        scores = self.score_net(
            camera_descriptors
        ).squeeze(-1)

        # =================================================
        # 3. Mask unavailable cameras
        # =================================================
        if camera_mask is not None:
            scores = scores.masked_fill(
                camera_mask == 0,
                -1e9
            )

        # =================================================
        # 4. Camera attention weights
        # =================================================
        # [B, K]
        weights = F.softmax(scores, dim=1)

        # =================================================
        # 5. Weighted fusion of camera tokens
        # =================================================
        # weights:
        # [B, K] -> [B, K, 1, 1]
        #
        # camera_tokens:
        # [B, K, N, D]
        #
        # fused_tokens:
        # [B, N, D]
        # =================================================
        fused_tokens = (
            camera_tokens
            * weights.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1)

        # =================================================
        # 6. Global token pooling
        # =================================================
        # [B, N, D] -> [B, D]
        pooled_tokens = fused_tokens.mean(dim=1)

        # =================================================
        # 7. Classification
        # =================================================
        # [B, D] -> [B, num_classes]
        logits = self.classifier(pooled_tokens)

        return fused_tokens, logits


class VideoMAEActivityClassifierPretrained(nn.Module):
    def __init__(
        self,
        checkpoint_path,
        num_classes=4,
        num_frames=16,
        freeze_backbone=True,
    ):
        super().__init__()

        # Build backbone
        self.backbone = vit_base_patch16_224(
            num_classes=710,  # original pretraining classes
            all_frames=num_frames,
            tubelet_size=2,
        )

        # Load pretrained weights
        print("Loading checkpoint...")
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        for model_key in ["model", "module"]:
            if model_key in ckpt:
                ckpt = ckpt[model_key]
                break

        self.backbone.load_state_dict(ckpt, strict=True)

        # Get feature dimension
        embed_dim = self.backbone.head.in_features

        # Replace classification head
        self.backbone.head = nn.Identity()

        # New activity classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_classes),
        )

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        x: [B, C, T, H, W]
        """


        if self.freeze_backbone:
            with torch.no_grad():
                tokens = self.backbone.forward_tokens(x)
        else:
            tokens = self.backbone.forward_tokens(x)
        features = self.backbone(x)   # [B, D]
        logits = self.classifier(features).sigmoid()

        return tokens, logits

class VideoMAEv2Classifier(nn.Module):

    def __init__(
        self,
        num_classes,
        model_name="OpenGVLab/VideoMAEv2-Base",
        num_frames=16,
        train_last_n_blocks=4,
        dropout=0.1
    ):
        super().__init__()

        self.num_frames = num_frames

        # =====================================================
        # LOAD PRETRAINED BACKBONE
        # =====================================================
        self.backbone = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        

        hidden_dim = self.backbone.config.model_config["embed_dim"]

        # =====================================================
        # FREEZE ALL PARAMETERS FIRST
        # =====================================================
        for p in self.backbone.parameters():
            p.requires_grad = False

        # =====================================================
        # UNFREEZE LAST TRANSFORMER BLOCKS
        # =====================================================
        #
        # Most VideoMAE implementations store blocks as:
        # self.backbone.blocks
        #
        # or
        # self.backbone.encoder.layer
        #
        # We handle both.
        #
        # =====================================================
        if hasattr(self.backbone, "blocks"):
                blocks = self.backbone.blocks

        elif hasattr(self.backbone, "model") and hasattr(self.backbone.model, "blocks"):
            blocks = self.backbone.model.blocks

        elif hasattr(self.backbone, "encoder"):

            if hasattr(self.backbone.encoder, "layer"):
                blocks = self.backbone.encoder.layer

            elif hasattr(self.backbone.encoder, "layers"):
                blocks = self.backbone.encoder.layers

            else:
                raise ValueError("Cannot find transformer blocks")

        else:
            raise ValueError("Cannot find transformer blocks")

              

        num_blocks = len(blocks)

        print(f"Total transformer blocks: {num_blocks}")

        # Unfreeze last N blocks
        for block in blocks[-train_last_n_blocks:]:

            for p in block.parameters():
                p.requires_grad = True

        # =====================================================
        # ALSO UNFREEZE FINAL NORM
        # =====================================================
        for name, module in self.backbone.named_modules():

            if "norm" in name.lower():

                for p in module.parameters():
                    p.requires_grad = True

        # =====================================================
        # CLASSIFICATION HEAD
        # =====================================================
        self.cls_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):

        """
        x:
            [B, T, 3, H, W]
        """

        B, T, C, H, W = x.shape

        assert T == self.num_frames

        # =====================================================
        # VideoMAE expects:
        # [B, C, T, H, W]
        # =====================================================
        x = x.permute(0, 2, 1, 3, 4)

        outputs = self.backbone(pixel_values=x)

        tokens = outputs.last_hidden_state

        # Global average pooling
        video_feat = tokens.mean(dim=1)

        logits = self.cls_head(video_feat).sigmoid()

        return video_feat, logits


class VideoClassificationLoss(nn.Module):
    """
    Loss function for video clip classification.

    Features:
    - Cross entropy classification loss
    - Optional label smoothing
    - Optional temporal feature regularization
    """

    def __init__(
        self,
        label_smoothing=0.1,
        temporal_consistency_weight=0.0
    ):
        super().__init__()

        self.temporal_consistency_weight = temporal_consistency_weight

        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )

    def forward(
        self,
        logits,
        labels,
        features=None
    ):
        """
        Args:
            logits:
                [B, num_classes]

            labels:
                [B]

            features:
                Optional temporal token features
                [B, N, D]

        Returns:
            total_loss
            loss_dict
        """

        # =====================================================
        # CLASSIFICATION LOSS
        # =====================================================
        cls_loss = self.ce_loss(logits, labels)

        total_loss = cls_loss

        loss_dict = {
            "cls_loss": cls_loss.item()
        }

        # =====================================================
        # OPTIONAL TEMPORAL SMOOTHNESS LOSS
        # Encourages neighboring temporal features
        # to be consistent
        # =====================================================
        if (
            features is not None and
            self.temporal_consistency_weight > 0
        ):

            # [B, N-1, D]
            feat1 = features[:, :-1]
            feat2 = features[:, 1:]

            temporal_loss = F.mse_loss(feat1, feat2)

            total_loss += (
                self.temporal_consistency_weight
                * temporal_loss
            )

            loss_dict["temporal_loss"] = temporal_loss.item()

        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict



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