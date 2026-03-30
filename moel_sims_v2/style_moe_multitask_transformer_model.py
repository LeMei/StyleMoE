import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class AttnPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        # x: [B, seq, dim]
        attn_scores = self.attn(x).squeeze(-1)  # [B, seq]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)
        # 防止in-place修改
        if mask is not None:
            only_pad = mask.all(dim=1)  # [B,]
            if only_pad.any():
                attn_weights = attn_weights.clone()
                attn_weights[only_pad] = 0.0
        out = (x * attn_weights.unsqueeze(-1)).sum(dim=1)
        return out


class AttnResidualMLPRouter(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, depth=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True, dropout=dropout)
        self.mlp = nn.ModuleList()
        for i in range(depth):
            self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(dropout))
        self.out_proj = nn.Linear(hidden_dim, num_experts)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-5)

    def forward(self, x):
        # x: [B, input_dim]
        x = self.input_proj(x)  # [B, hidden_dim]
        x_ = x.unsqueeze(1)     # [B, 1, hidden_dim] -> attention输入
        attn_out, _ = self.self_attn(x_, x_, x_)  # [B, 1, hidden_dim]
        attn_out = attn_out.squeeze(1)            # [B, hidden_dim]
        x = x + attn_out                         # 残差
        x = self.norm(x)
        for layer in self.mlp:
            x = layer(x)
        logits = self.out_proj(x)
        return logits

def modality_dropout(x, drop_prob, training):
    # x: [B, seq, dim] or [B, dim]
    if (not training) or (drop_prob <= 0.):
        return x
    device = x.device
    batch_size = x.shape[0]
    # 以概率drop_prob将每个样本mask为0
    mask = (torch.rand(batch_size, device=device) > drop_prob).float().unsqueeze(-1)
    # 针对2D/3D输入通用
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(1)
    return x * mask


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            # 在初始化时就将 alpha 转换为 tensor 并注册为 buffer
            # buffer 不会被视为模型参数，但会随模型移动 (例如 .to(device))
            if not isinstance(alpha, torch.Tensor): # 如果传入的是列表或numpy数组
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha_t', alpha) # 注册为alpha_t，避免与方法参数重名
        else:
            self.alpha_t = None # 或者 self.register_buffer('alpha_t', None) 但通常None不需要注册
        self.reduction = reduction

    def forward(self, input, target):
        ce = F.cross_entropy(input, target, reduction="none")
        pt = torch.exp(-ce)
        
        loss = ce # 默认损失是ce

        if self.alpha_t is not None:
            # 直接使用预先转换好的 self.alpha_t
            at = self.alpha_t[target] # self.alpha_t 已经是tensor了，并且在正确的device上
            loss = loss * at
            
        focal_loss = (1 - pt) ** self.gamma * loss # 使用更新后的loss (可能已被alpha加权)
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else: # "none"
            return focal_loss

class ModalityTransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim=128, num_heads=4, num_layers=2, dropout=0.1, use_attn_pool=True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim, eps=1e-5)
        self.use_attn_pool = use_attn_pool
        if use_attn_pool:
            self.pool = AttnPooling(model_dim)
        else:
            self.pool = None

    def forward(self, x):
        # x: [B, seq_len, input_dim]
        mask = (x.abs().sum(dim=-1) == 0)  # [B, seq_len]
        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)
        
        if self.use_attn_pool:
            out = self.pool(x, mask)
        else:
            # mean pooling（老方法，留着对比）
            valid_count = (~mask).sum(dim=1, keepdim=True).clamp(min=1)
            x_masked = x.masked_fill(mask.unsqueeze(-1), 0.0)
            out = x_masked.sum(dim=1) / valid_count
        return out


class CrossModalAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=out_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.query_proj = nn.Linear(q_dim, out_dim)
        self.kv_proj = nn.Linear(kv_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim, eps=1e-5)

    def forward(self, q, kv):
        # q: [B, 1, q_dim], kv: [B, 50, kv_dim]
        q_proj = self.query_proj(q)
        kv_proj = self.kv_proj(kv)
        out, _ = self.attn(q_proj, kv_proj, kv_proj)
        out = self.norm(out)
        return out.squeeze(1)  # [B, out_dim]

class StyleMoEMultiTaskTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # === config参数 ===
        text_dim = config["model"]["text_dim"]
        audio_dim = config["model"]["audio_dim"]
        vision_dim = config["model"]["visual_dim"]
        num_classes = config["model"]["num_classes"] # This should be 2 for binary classification
        hidden_dim = config["model"]["hidden_dim"] # 主融合后的隐藏维度
        style_dim = config["model"]["style_dim"]
        router_hidden_dim = config["model"]["router_hidden_dim"]
        router_depth = config["model"].get("router_depth", 2)
        cls_hidden_dim = config["model"].get("cls_hidden_dim", 128) # 主分类头隐藏维度
        cls_dropout = config["model"].get("cls_dropout", 0.5)     # 主分类头dropout

        self.dropout_prob = config["model"]["dropout_prob"]
        self.lambda_style = config["model"]["lambda_style"]
        # [REMOVED] self.contrastive_margin = config["model"]["contrastive_margin"]
        
        if "cluster_path" in config["paths"]:
            centers_path = config["paths"]["cluster_path"]
            if centers_path and os.path.exists(centers_path): # Check if path is not None/empty and exists
                 centers_data = np.load(centers_path)
                 self.centers = torch.tensor(centers_data, dtype=torch.float32)
            else:
                 print(f"[聚类初始化警告] cluster_path '{centers_path}' 未找到或为空, 跳过聚类初始化。")
                 self.centers = None
        else:
            self.centers = None


        # === Transformer编码 ===
        # 假设这些编码器的 model_dim 分别是其输出维度
        self.text_encoder_dim = config["model"].get("text_encoder_model_dim", 128)
        self.audio_encoder_dim = config["model"].get("audio_encoder_model_dim", 32)
        self.vision_encoder_dim = config["model"].get("vision_encoder_model_dim", 32)
        
        self.text_encoder = ModalityTransformerEncoder(input_dim=text_dim, model_dim=self.text_encoder_dim)
        self.audio_encoder = ModalityTransformerEncoder(input_dim=audio_dim, model_dim=self.audio_encoder_dim)
        self.vision_encoder = ModalityTransformerEncoder(input_dim=vision_dim, model_dim=self.vision_encoder_dim)

        # === 风格编码 ===
        self.text_to_style = nn.Linear(self.text_encoder_dim, style_dim)
        self.audio_to_style = nn.Linear(self.audio_encoder_dim, style_dim)
        self.vision_to_style = nn.Linear(self.vision_encoder_dim, style_dim)
        
        # [REMOVED] self.criterion_contrastive = nn.CosineEmbeddingLoss(margin=self.contrastive_margin)
        
        # [ADDED] InfoNCE/CLIP learnable temperature parameter
        # Initialized to log(1/0.07) ~ 2.659, the value used in CLIP
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


        # === 单模态专家 ===
        self.text_expert = nn.Sequential(nn.Linear(self.text_encoder_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim, eps=1e-5))
        self.audio_expert = nn.Sequential(nn.Linear(self.audio_encoder_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim, eps=1e-5))
        self.vision_expert = nn.Sequential(nn.Linear(self.vision_encoder_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim, eps=1e-5))
        
        # === 跨模态交互+专家 ===
        cross_modal_out_dim = config["model"].get("cross_modal_out_dim", 64)
        self.cross_text_audio = CrossModalAttention(
            q_dim=self.text_encoder_dim,
            kv_dim=audio_dim, # Use raw audio_dim for KV if attending to raw sequences
            out_dim=cross_modal_out_dim
        )
        self.cross_text_vision = CrossModalAttention(
            q_dim=self.text_encoder_dim,
            kv_dim=vision_dim, # Use raw visual_dim for KV
            out_dim=cross_modal_out_dim
        )
        self.cross_audio_expert = nn.Sequential(nn.Linear(cross_modal_out_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim, eps=1e-5))
        self.cross_vision_expert = nn.Sequential(nn.Linear(cross_modal_out_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim, eps=1e-5))

        # === Router (MoE门控) ===
        input_dim_router = self.text_encoder_dim + self.audio_encoder_dim + self.vision_encoder_dim + cross_modal_out_dim + cross_modal_out_dim
        self.router = AttnResidualMLPRouter(
            input_dim=input_dim_router,
            hidden_dim=router_hidden_dim,
            num_experts=5, 
            depth=router_depth,
            dropout=config["model"]["router_dropout"]
        )
        print(f"[Router] dropout: {config['model']['router_dropout']}, depth: {router_depth}, hidden_dim: {router_hidden_dim}")

        # === 主多任务输出头 ===
        reg_head_hidden_dim = config["model"].get("reg_hidden_dim", hidden_dim) # Default to hidden_dim if not specified
        reg_head_dropout = config["model"].get("reg_dropout", 0.2)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, reg_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(reg_head_dropout),
            nn.Linear(reg_head_hidden_dim, 1)
        )
        # Main classification head (for 'non0' task)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, cls_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden_dim, num_classes) # num_classes should be 2
        )
        
        # === 新增：为 'has0' 任务设计的专用分类头 ===
        # 我们可以复用 cls_hidden_dim 和 cls_dropout，或者在config中定义新的
        # has0_cls_hidden_dim = config["model"].get("has0_cls_hidden_dim", cls_hidden_dim)
        # has0_cls_dropout = config["model"].get("has0_cls_dropout", cls_dropout)
        self.cls_head_has0 = nn.Sequential(
            nn.Linear(hidden_dim, cls_hidden_dim), # Reusing cls_hidden_dim for simplicity
            nn.ReLU(),
            nn.Dropout(cls_dropout),             # Reusing cls_dropout for simplicity
            nn.Linear(cls_hidden_dim, num_classes) # num_classes should be 2
        )
        # =========================================

        # === 辅助单模态输出头 ===
        aux_cls_hid_dim_ratio = config["model"].get("aux_cls_hidden_dim_ratio", 0.5) 
        aux_cls_dropout_ratio = config["model"].get("aux_cls_dropout_ratio", 1.0)   
        
        aux_cls_hidden_dim_actual = max(16, int(cls_hidden_dim * aux_cls_hid_dim_ratio)) 
        aux_cls_dropout_actual = cls_dropout * aux_cls_dropout_ratio

        self.aux_text_reg_head = nn.Linear(self.text_encoder_dim, 1)
        self.aux_text_cls_head = nn.Sequential(
            nn.Linear(self.text_encoder_dim, aux_cls_hidden_dim_actual), nn.ReLU(),
            nn.Dropout(aux_cls_dropout_actual), nn.Linear(aux_cls_hidden_dim_actual, num_classes)
        )
        self.aux_audio_reg_head = nn.Linear(self.audio_encoder_dim, 1)
        self.aux_audio_cls_head = nn.Sequential(
            nn.Linear(self.audio_encoder_dim, max(16, aux_cls_hidden_dim_actual // 2)), nn.ReLU(), # ensure min_dim
            nn.Dropout(aux_cls_dropout_actual), nn.Linear(max(16, aux_cls_hidden_dim_actual // 2), num_classes)
        )
        self.aux_vision_reg_head = nn.Linear(self.vision_encoder_dim, 1)
        self.aux_vision_cls_head = nn.Sequential(
            nn.Linear(self.vision_encoder_dim, max(16, aux_cls_hidden_dim_actual // 2)), nn.ReLU(), # ensure min_dim
            nn.Dropout(aux_cls_dropout_actual), nn.Linear(max(16, aux_cls_hidden_dim_actual // 2), num_classes)
        )
        
        if self.centers is not None:
            centers_np = self.centers.detach().cpu().numpy()
            print(f"[聚类初始化] centers_np shape: {centers_np.shape}")
            experts_list = [
                self.text_expert[0], self.audio_expert[0], self.vision_expert[0],
                self.cross_audio_expert[0], self.cross_vision_expert[0]
            ]
            for i, expert_layer in enumerate(experts_list):
                if i < centers_np.shape[0]:
                    center_vec = centers_np[i]
                    new_bias = np.zeros_like(expert_layer.bias.data.cpu().numpy())
                    fill_dim = min(len(center_vec), len(new_bias))
                    new_bias[:fill_dim] = center_vec[:fill_dim]
                    with torch.no_grad():
                        expert_layer.bias.copy_(torch.tensor(new_bias, dtype=expert_layer.bias.dtype))
                    print(f"[聚类初始化] expert {i} bias 用聚类中心前{fill_dim}维初始化 (目标bias长度={len(new_bias)}, 中心向量长度={len(center_vec)})")
                else:
                    print(f"[聚类初始化] expert {i} 没有对应聚类中心，跳过")
        else:
            print("[聚类初始化] self.centers is None，跳过聚类初始化")


    def forward(self, text, audio, vision, return_expert_outputs=False):
        text_dropped = modality_dropout(text, self.dropout_prob, self.training)
        audio_dropped = modality_dropout(audio, self.dropout_prob, self.training)
        vision_dropped = modality_dropout(vision, self.dropout_prob, self.training)

        t_feat = self.text_encoder(text_dropped)
        a_feat = self.audio_encoder(audio_dropped)
        v_feat = self.vision_encoder(vision_dropped)
        
        aux_text_reg = self.aux_text_reg_head(t_feat)
        aux_text_cls_logits = self.aux_text_cls_head(t_feat)
        aux_audio_reg = self.aux_audio_reg_head(a_feat)
        aux_audio_cls_logits = self.aux_audio_cls_head(a_feat)
        aux_vision_reg = self.aux_vision_reg_head(v_feat)
        aux_vision_cls_logits = self.aux_vision_cls_head(v_feat)

        text_style = self.text_to_style(t_feat)
        audio_style = self.audio_to_style(a_feat)
        vision_style = self.vision_to_style(v_feat)
        
        # [MODIFIED] InfoNCE/CLIP-style contrastive loss block
        
        # 1. L2-normalize features
        text_style = F.normalize(text_style, p=2, dim=-1)
        audio_style = F.normalize(audio_style, p=2, dim=-1)
        vision_style = F.normalize(vision_style, p=2, dim=-1)

        # 2. Get learned temperature (logit_scale)
        # Clamp to prevent overflow/instability, as in official CLIP
        logit_scale = self.logit_scale.exp().clamp(max=100)
        
        # 3. Calculate cosine similarity matrices
        # [B, B] similarity matrix for (Text, Audio)
        logits_t_a = (text_style @ audio_style.T) * logit_scale
        # [B, B] similarity matrix for (Text, Vision)
        logits_t_v = (text_style @ vision_style.T) * logit_scale
        
        # 4. Create ground-truth labels (diagonal indices)
        batch_size = text_style.shape[0]
        # Check for batch_size > 0 to avoid error on empty batches
        if batch_size == 0:
            # If batch is empty, set loss to 0 and skip calculation
            style_loss_val = torch.tensor(0.0, device=t_feat.device)
        else:
            labels = torch.arange(batch_size, device=text_style.device) # [0, 1, 2, ..., B-1]

            # 5. Calculate symmetric cross-entropy loss
            loss_t_a = F.cross_entropy(logits_t_a, labels)    # Text-to-Audio
            loss_a_t = F.cross_entropy(logits_t_a.T, labels)  # Audio-to-Text
            
            loss_t_v = F.cross_entropy(logits_t_v, labels)    # Text-to-Vision
            loss_v_t = F.cross_entropy(logits_t_v.T, labels)  # Vision-to-Text

            # 6. Average the four losses
            contrastive_loss = (loss_t_a + loss_a_t + loss_t_v + loss_v_t) / 4.0

            # 7. Apply weighting (maintaining original logic from config)
            # We no longer use mse_loss
            style_loss_val = self.lambda_style * contrastive_loss
        
        # [END MODIFIED BLOCK]


        t_out = self.text_expert(t_feat)
        a_out = self.audio_expert(a_feat)
        v_out = self.vision_expert(v_feat)

        # Pass original (pre-dropout, pre-encoder) sequences to cross-modal attention if that's the design.
        # Or pass t_feat, audio_dropped (sequences), vision_dropped (sequences) if Q is encoded and K/V are sequences.
        # Current CrossModalAttention expects K/V to be sequences that it projects.
        # Q is t_feat (pooled).
        t2a = self.cross_text_audio(t_feat.unsqueeze(1), audio_dropped) # K/V are raw sequences
        t2v = self.cross_text_vision(t_feat.unsqueeze(1), vision_dropped) # K/V are raw sequences

        t2a_out = self.cross_audio_expert(t2a)
        t2v_out = self.cross_vision_expert(t2v)

        if return_expert_outputs:
            # 专用于 t-SNE 数据导出，只返回专家输出
            return t_out, a_out, v_out, t2a_out, t2v_out

        experts = torch.stack([t_out, a_out, v_out, t2a_out, t2v_out], dim=1)
        router_input = torch.cat([t_feat, a_feat, v_feat, t2a, t2v], dim=-1)
        weights = F.softmax(self.router(router_input), dim=-1).unsqueeze(-1)
        fused = torch.sum(weights * experts, dim=1)
        
        main_regression = self.reg_head(fused)
        main_classification_logits = self.cls_head(fused) # For 'non0' task
        
        # === 新增：获取 'has0' 专用分类头的输出 ===
        has0_classification_logits = self.cls_head_has0(fused) # For 'has0' task
        # =======================================

        return (main_regression, main_classification_logits, style_loss_val,
                aux_text_reg, aux_text_cls_logits,
                aux_audio_reg, aux_audio_cls_logits,
                aux_vision_reg, aux_vision_cls_logits,
                has0_classification_logits) # 新增的输出