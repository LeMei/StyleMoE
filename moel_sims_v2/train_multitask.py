import argparse
import torch
import yaml
import pickle
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, accuracy_score as sk_accuracy_score, f1_score
from scipy.stats import pearsonr
from transformers import get_cosine_schedule_with_warmup
# 确保从正确路径导入模型
from style_moe_multitask_transformer_model import StyleMoEMultiTaskTransformer, FocalLoss # FocalLoss在这里用于主分类和辅助分类

import random
import copy # For deepcopying config if needed for Optuna later

seed = 17 # 或者你选择的任何整数
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# === multiclass_acc 函数 (保持不变) ===
def multiclass_acc(y_pred_reg, y_true_reg, clip_range=None):
    if clip_range:
        y_pred_reg = np.clip(y_pred_reg, clip_range[0], clip_range[1])
        y_true_reg = np.clip(y_true_reg, clip_range[0], clip_range[1])
    y_pred_class = np.round(y_pred_reg).astype(int)
    y_true_class = np.round(y_true_reg).astype(int)
    if len(y_true_class) == 0: 
        return np.nan
    return sk_accuracy_score(y_true_class, y_pred_class)

# === 配置读取 (保持不变) ===
parser = argparse.ArgumentParser(description="Train a multi-task model.")
parser.add_argument('--config_path', type=str, default='config.yaml', help='Path to the configuration file.')
args = parser.parse_args()

# script_dir = os.path.dirname(os.path.abspath(__file__)) # 这行可以删除或注释掉
# config_path = os.path.join(script_dir, "config.yaml") # 这行可以删除或注释掉
with open(args.config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print(f"成功加载配置文件: {args.config_path}")


if config["model"].get("num_classes") != 2:
    print(f"警告: 配置文件中的 num_classes ({config['model'].get('num_classes')}) 不是 2。")
    print("此脚本被修改为进行二分类。将强制设置 num_classes = 2。")
    config["model"]["num_classes"] = 2
    if "focal_alpha" not in config["training"] or len(config["training"]["focal_alpha"]) != 2:
        print("警告: focal_alpha 未设置或长度不为2，将使用默认 [0.5, 0.5]。")
        config["training"]["focal_alpha"] = [0.5, 0.5]

POSITIVE_THRESHOLD_FOR_BINARY_NON_NEUTRAL = config.get("dataset_specifics", {}).get("positive_threshold_binary_non_neutral", 0.00001) 
NEGATIVE_THRESHOLD_FOR_BINARY_NON_NEUTRAL = config.get("dataset_specifics", {}).get("negative_threshold_binary_non_neutral", -0.00001)
POSITIVE_THRESHOLD_FOR_BINARY_HAS0 = config.get("dataset_specifics", {}).get("positive_threshold_binary_has0", 0.0) 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = config["paths"]["data_dir"]
dataset_name = os.path.basename(data_dir) 
experiment_name = config.get("experiment_name", f"{dataset_name}_multiloss_comprehensive_metrics_has0head") # 更新实验名
output_dir = os.path.join(config["paths"]["output_dir"], experiment_name)
os.makedirs(output_dir, exist_ok=True)

data_file_path = os.path.join(data_dir, f"unaligned.pkl") # 假设使用了BERT特征的数据集
if not os.path.exists(data_file_path):
    # Fallback or specific filename if convention is different
    data_file_path_alt = os.path.join(data_dir, "unaligned.pkl") 
    if os.path.exists(data_file_path_alt):
        data_file_path = data_file_path_alt
    else:
        print(f"错误: 数据文件 {data_file_path} 或备选路径均未找到!")
        exit(1)
print(f"加载数据从: {data_file_path}")
with open(data_file_path, "rb") as f:
    data = pickle.load(f)

# === 数据预处理函数 (valid_seq_mask, clean_inf_nan, preprocess - 保持不变) ===
def valid_seq_mask(x):
    return (x.abs().sum(dim=-1) > 0).sum(dim=1) >= 1

def clean_inf_nan(tensor):
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    return tensor

def preprocess(split_name, p_thresh_non_neutral, n_thresh_non_neutral, p_thresh_has0):
    split_data = data[split_name]
    text_data = split_data.get('text', split_data.get('text', split_data.get('words', split_data.get('text_features'))))
    print(f"原始加载的 text_data ({split_name}) 的类型: {type(text_data)}")
    if hasattr(text_data, 'shape'):
        print(f"原始加载的 text_data ({split_name}) 的形状: {text_data.shape}")
    audio_data = split_data.get('audio', split_data.get('audio_features'))
    vision_data = split_data.get('vision', split_data.get('vision_features'))
    reg_labels_data = split_data.get('regression_labels', split_data.get('labels'))

    text = torch.tensor(text_data, dtype=torch.float32)
    print(f"转换为 tensor 后 text ({split_name}) 的形状: {text.shape}") # 重点关注这个形状
    audio = torch.tensor(audio_data, dtype=torch.float32)
    vision = torch.tensor(vision_data, dtype=torch.float32)
    original_reg_labels = torch.tensor(reg_labels_data, dtype=torch.float32)

    text = clean_inf_nan(text)
    audio = clean_inf_nan(audio)
    vision = clean_inf_nan(vision)

    if original_reg_labels.ndim == 3:
        original_reg_labels = original_reg_labels.squeeze(-1)
    original_reg_labels = original_reg_labels.view(-1, 1)

    base_valid_mask = valid_seq_mask(text) & valid_seq_mask(audio) & valid_seq_mask(vision)
    
    text_out = text[base_valid_mask]
    audio_out = audio[base_valid_mask]
    vision_out = vision[base_valid_mask]
    reg_labels_out = original_reg_labels[base_valid_mask]

    if reg_labels_out.numel() == 0:
        print(f"警告: Split '{split_name}' 基础过滤后无样本。")
        # Determine shape from original tensors before filtering if possible, or use typical shapes
        text_shape_orig = text_data.shape if hasattr(text_data, 'shape') else (0, config["model"]["text_dim"]) 
        audio_shape_orig = audio_data.shape if hasattr(audio_data, 'shape') else (0, config["model"]["audio_dim"])
        vision_shape_orig = vision_data.shape if hasattr(vision_data, 'shape') else (0, config["model"]["visual_dim"])
        
        # Ensure correct number of dimensions for sequence data if applicable (e.g., text might be [N, seq_len, feat_dim])
        # For simplicity, assuming features are [N, feat_dim] after potential initial processing (like mean in init_cluster)
        # If your features are [N, Seq, Dim], adjust text_out, audio_out, vision_out empty tensor shapes
        # Example for text: torch.empty(0, text_shape_orig[1] if len(text_shape_orig)>1 else 0 , text_shape_orig[2] if len(text_shape_orig)>2 else 0)
        # Assuming pooled features for simplicity here:
        return torch.empty(0, text_shape_orig[-1]), \
               torch.empty(0, audio_shape_orig[-1]), \
               torch.empty(0, vision_shape_orig[-1]), \
               torch.empty(0,1), \
               torch.empty(0, dtype=torch.long), \
               torch.empty(0, dtype=torch.bool), \
               torch.empty(0, dtype=torch.long)

    reg_labels_squeezed = reg_labels_out.squeeze()
    
    is_positive_non_neutral = reg_labels_squeezed > p_thresh_non_neutral
    is_negative_non_neutral = reg_labels_squeezed < n_thresh_non_neutral
    non_neutral_mask_out = is_positive_non_neutral | is_negative_non_neutral
    binary_cls_labels_non0_out = torch.zeros(reg_labels_out.shape[0], dtype=torch.long)
    binary_cls_labels_non0_out[is_positive_non_neutral] = 1 

    binary_cls_labels_has0_out = torch.zeros(reg_labels_out.shape[0], dtype=torch.long)
    binary_cls_labels_has0_out[reg_labels_squeezed > p_thresh_has0] = 1 

    print(f"Split '{split_name}': Total valid samples {len(text_out)}, Non-neutral for 'non0' classification {non_neutral_mask_out.sum().item()}")
    return text_out, audio_out, vision_out, reg_labels_out, binary_cls_labels_non0_out, non_neutral_mask_out, binary_cls_labels_has0_out
  
train_text, train_audio, train_vision, train_reg_y, train_cls_y_non0, train_non_neutral_mask, train_cls_y_has0 = preprocess(
    "train", POSITIVE_THRESHOLD_FOR_BINARY_NON_NEUTRAL, NEGATIVE_THRESHOLD_FOR_BINARY_NON_NEUTRAL, POSITIVE_THRESHOLD_FOR_BINARY_HAS0
)
val_text, val_audio, val_vision, val_reg_y, val_cls_y_non0, val_non_neutral_mask, val_cls_y_has0 = preprocess(
    "valid", POSITIVE_THRESHOLD_FOR_BINARY_NON_NEUTRAL, NEGATIVE_THRESHOLD_FOR_BINARY_NON_NEUTRAL, POSITIVE_THRESHOLD_FOR_BINARY_HAS0
)

if train_text.numel() == 0 or val_text.numel() == 0:
    print("错误: 训练集或验证集在预处理后为空。")
    exit()

train_loader = DataLoader(TensorDataset(train_text, train_audio, train_vision, train_reg_y, train_cls_y_non0, train_non_neutral_mask, train_cls_y_has0),
                          batch_size=config["training"]["batch_size"], shuffle=True, pin_memory=device.type=='cuda')
val_loader = DataLoader(TensorDataset(val_text, val_audio, val_vision, val_reg_y, val_cls_y_non0, val_non_neutral_mask, val_cls_y_has0),
                        batch_size=config["training"]["batch_size"], pin_memory=device.type=='cuda')

model = StyleMoEMultiTaskTransformer(config).to(device)
criterion_reg = torch.nn.SmoothL1Loss()
focal_alpha_tensor = torch.tensor(config["training"]["focal_alpha"], dtype=torch.float32).to(device)
# Criterion for main 'non0' classification and auxiliary classifications
criterion_cls_non0_aux = FocalLoss(gamma=config["training"]["focal_gamma"], alpha=focal_alpha_tensor)
criterion_cls_non0_aux.to(device)

# === 新增：为 'has0' 专用头创建损失函数 ===
# 你可以为 'has0' 头使用不同的 alpha 和 gamma，或者复用 'non0' 的设置
# 如果复用，确保 focal_alpha (类别权重) 对 'has0' 任务也合适
# 例如，如果 'has0' 任务中正负类别比例与 'non0' 不同，可能需要不同的 alpha
# 这里我们先复用，但你可以考虑在 config.yaml 中为 has0 定义独立的 focal_alpha_has0 和 focal_gamma_has0
has0_focal_alpha = config["training"].get("focal_alpha_has0", config["training"]["focal_alpha"])
has0_focal_gamma = config["training"].get("focal_gamma_has0", config["training"]["focal_gamma"])
has0_focal_alpha_tensor = torch.tensor(has0_focal_alpha, dtype=torch.float32).to(device)
criterion_cls_has0_dedicated = FocalLoss(gamma=has0_focal_gamma, alpha=has0_focal_alpha_tensor)
criterion_cls_has0_dedicated.to(device)
# =======================================

optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
num_total_steps_train = len(train_loader) * config["training"]["epochs"]
num_warmup_steps_train = int(num_total_steps_train * 0.1) if len(train_loader) > 10 else max(1, len(train_loader))
if num_total_steps_train == 0 : num_total_steps_train = 1 # Avoid division by zero if train_loader is empty after all
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps_train, num_training_steps=num_total_steps_train)

best_combined_score = -float('inf')
patience = 0
log_file_main_train = os.path.join(output_dir, "train_val_log_comprehensive_has0head.txt") # 更新日志文件名
# Save config to output_dir for this run
with open(os.path.join(output_dir, "config_this_run.yaml"), "w", encoding="utf-8") as f_cfg:
    yaml.dump(config, f_cfg)

aux_weights = config["training"].get("aux_loss_weights", {})
loss_weight_has0_cls_val = config["training"].get("loss_weight_has0_cls", 0.75) # 从config获取权重

for epoch in range(config["training"]["epochs"]):
    model.train()
    epoch_train_loss_sum = 0
    num_train_batches_epoch = 0
    for t, a, v, reg_y_batch, cls_y_non0_batch, non_neutral_mask_batch, cls_y_has0_batch in train_loader:
        batch_mask_valid_input = (t.abs().sum(dim=(1,2)) + a.abs().sum(dim=(1,2)) + v.abs().sum(dim=(1,2))) > 0
        if not hasattr(t, 'shape') or t.shape[0] == 0 or batch_mask_valid_input.sum() == 0 : continue
        
        t_m,a_m,v_m,reg_y_m,cls_y_non0_m,non_neutral_m, cls_y_has0_m = \
            [x[batch_mask_valid_input] for x in [t,a,v,reg_y_batch,cls_y_non0_batch,non_neutral_mask_batch, cls_y_has0_batch]]
        if t_m.size(0) == 0: continue

        t_d,a_d,v_d,reg_y_d,cls_y_non0_d,non_neutral_d, cls_y_has0_d = \
            [x.to(device) for x in [t_m,a_m,v_m,reg_y_m,cls_y_non0_m,non_neutral_m, cls_y_has0_m]]

        outputs = model(t_d, a_d, v_d) # Model now returns 10 values
        pred_reg, pred_cls_logits_non0, style_loss = outputs[0], outputs[1], outputs[2]
        aux_text_reg, aux_text_cls_logits = outputs[3], outputs[4]
        aux_audio_reg, aux_audio_cls_logits = outputs[5], outputs[6]
        aux_vision_reg, aux_vision_cls_logits = outputs[7], outputs[8]
        pred_cls_logits_has0_dedicated = outputs[9] # 新增的 'has0' 专用头输出
        
        nan_detected = False
        # 更新 NaN 检查以包含新的输出
        all_outputs_for_nan_check = [pred_reg, pred_cls_logits_non0, style_loss, 
                                     aux_text_reg, aux_text_cls_logits,
                                     aux_audio_reg, aux_audio_cls_logits,
                                     aux_vision_reg, aux_vision_cls_logits,
                                     pred_cls_logits_has0_dedicated] # 加入检查
        for out_tensor in all_outputs_for_nan_check:
            if out_tensor is not None and torch.isnan(out_tensor).any():
                nan_detected = True; break
        if nan_detected: 
            print(f"E{epoch+1} TRAIN: NaN in model output. Skipping batch.")
            with open(log_file_main_train, "a", encoding="utf-8") as f_log_err:
                f_log_err.write(f"E{epoch+1} TRAIN: NaN in model output. Skipping batch.\n")
            continue
            
        loss_r = criterion_reg(pred_reg, reg_y_d)
        
        # 主 'non0' 分类损失 (只在 "non0" 样本上计算)
        loss_c_non0 = torch.tensor(0.0, device=device)
        if non_neutral_d.sum() > 0:
            pred_cls_logits_non0_filtered = pred_cls_logits_non0[non_neutral_d]
            cls_y_binary_d_non0_filtered = cls_y_non0_d[non_neutral_d]
            if pred_cls_logits_non0_filtered.size(0) > 0:
                 loss_c_non0 = criterion_cls_non0_aux(pred_cls_logits_non0_filtered, cls_y_binary_d_non0_filtered)
        
        # === 新增: 计算 'has0' 专用头的损失 ===
        # 这个损失在所有样本上计算，使用 cls_y_has0_d 作为标签
        loss_c_has0_dedicated = torch.tensor(0.0, device=device)
        if pred_cls_logits_has0_dedicated.size(0) > 0: # Ensure batch is not empty
            loss_c_has0_dedicated = criterion_cls_has0_dedicated(pred_cls_logits_has0_dedicated, cls_y_has0_d)
        # =====================================
        
        # 辅助损失 (分类部分在 "non0" 样本上计算)
        loss_r_text_aux = criterion_reg(aux_text_reg, reg_y_d)
        loss_c_text_aux = torch.tensor(0.0, device=device)
        if non_neutral_d.sum() > 0:
            if aux_text_cls_logits[non_neutral_d].size(0) > 0 :
                 loss_c_text_aux = criterion_cls_non0_aux(aux_text_cls_logits[non_neutral_d], cls_y_non0_d[non_neutral_d])

        loss_r_audio_aux = criterion_reg(aux_audio_reg, reg_y_d)
        loss_c_audio_aux = torch.tensor(0.0, device=device)
        if non_neutral_d.sum() > 0:
            if aux_audio_cls_logits[non_neutral_d].size(0) > 0:
                loss_c_audio_aux = criterion_cls_non0_aux(aux_audio_cls_logits[non_neutral_d], cls_y_non0_d[non_neutral_d])

        loss_r_vision_aux = criterion_reg(aux_vision_reg, reg_y_d)
        loss_c_vision_aux = torch.tensor(0.0, device=device)
        if non_neutral_d.sum() > 0:
            if aux_vision_cls_logits[non_neutral_d].size(0) > 0:
                loss_c_vision_aux = criterion_cls_non0_aux(aux_vision_cls_logits[non_neutral_d], cls_y_non0_d[non_neutral_d])

        # 更新损失 NaN 检查
        loss_components = [loss_r, loss_c_non0, style_loss, loss_c_has0_dedicated, # 加入新损失
                           loss_r_text_aux, loss_c_text_aux, 
                           loss_r_audio_aux, loss_c_audio_aux, 
                           loss_r_vision_aux, loss_c_vision_aux]
        if any(l is not None and torch.isnan(l).any() for l in loss_components):
            print(f"E{epoch+1} TRAIN: NaN in loss components. Skipping batch.")
            with open(log_file_main_train, "a", encoding="utf-8") as f_log_err:
                f_log_err.write(f"E{epoch+1} TRAIN: NaN in loss components. Skipping batch.\n")
            continue

        total_loss = (config["training"].get("loss_reg", 1.0) * loss_r +
                      config["training"]["loss_alpha"] * loss_c_non0 + 
                      config["training"]["loss_beta"] * style_loss +
                      loss_weight_has0_cls_val * loss_c_has0_dedicated) # 加入新损失的加权值
        
        total_loss += aux_weights.get("text_reg", 0.0) * loss_r_text_aux
        total_loss += aux_weights.get("text_cls", 0.0) * loss_c_text_aux
        total_loss += aux_weights.get("audio_reg", 0.0) * loss_r_audio_aux
        total_loss += aux_weights.get("audio_cls", 0.0) * loss_c_audio_aux
        total_loss += aux_weights.get("vision_reg", 0.0) * loss_r_vision_aux
        total_loss += aux_weights.get("vision_cls", 0.0) * loss_c_vision_aux
        
        if torch.isnan(total_loss).any():
            print(f"E{epoch+1} TRAIN: NaN in total_loss. Skipping batch.")
            with open(log_file_main_train, "a", encoding="utf-8") as f_log_err:
                f_log_err.write(f"E{epoch+1} TRAIN: NaN in total_loss. Skipping batch.\n")
            continue

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        epoch_train_loss_sum += total_loss.item()
        num_train_batches_epoch += 1
    
    avg_epoch_train_loss = epoch_train_loss_sum / num_train_batches_epoch if num_train_batches_epoch > 0 else float('nan')

    # === 验证 ===
    model.eval()
    all_reg_preds_val, all_reg_labels_val = [], []
    non0_cls_preds_val, non0_cls_labels_val = [], [] # For 'non0' task from main cls_head
    has0_cls_preds_val_dedicated, has0_cls_labels_val_dedicated = [], [] # For 'has0' task from dedicated cls_head_has0
    
    with torch.no_grad():
        for t_val, a_val, v_val, reg_y_val_b, cls_y_non0_val_b, non_neutral_mask_val_b, cls_y_has0_val_b in val_loader:
            if not hasattr(t_val, 'shape') or t_val.shape[0] == 0: continue
            
            t_v_d,a_v_d,v_v_d,reg_y_v_d,cls_y_non0_v_d,non_neutral_v_d, cls_y_has0_v_d = \
                [x.to(device) for x in [t_val,a_val,v_val,reg_y_val_b,cls_y_non0_val_b,non_neutral_mask_val_b, cls_y_has0_val_b]]

            outputs_val = model(t_v_d, a_v_d, v_v_d)
            pred_reg_val = outputs_val[0]
            pred_cls_binary_non0_val_logits = outputs_val[1] # Main head for 'non0'
            # pred_style_loss_val = outputs_val[2] # Not needed for validation metrics
            # aux outputs ... outputs_val[3] to outputs_val[8]
            pred_cls_binary_has0_dedicated_val_logits = outputs_val[9] # Dedicated head for 'has0'

            if torch.isnan(pred_reg_val).any() or \
               torch.isnan(pred_cls_binary_non0_val_logits).any() or \
               torch.isnan(pred_cls_binary_has0_dedicated_val_logits).any():
                print(f"E{epoch+1} VAL: NaN in model output. Skipping batch.")
                with open(log_file_main_train, "a", encoding="utf-8") as f_log_err:
                    f_log_err.write(f"E{epoch+1} VAL: NaN in model output. Skipping batch.\n")
                continue

            all_reg_preds_val.extend(pred_reg_val.cpu().numpy())
            all_reg_labels_val.extend(reg_y_v_d.cpu().numpy())

            # "non0" 二分类 (来自主 cls_head)
            if non_neutral_v_d.sum() > 0:
                pred_cls_logits_nn_val = pred_cls_binary_non0_val_logits[non_neutral_v_d]
                cls_y_binary_nn_val = cls_y_non0_v_d[non_neutral_v_d]
                if pred_cls_logits_nn_val.size(0) > 0:
                    non0_cls_preds_val.extend(pred_cls_logits_nn_val.argmax(dim=1).cpu().numpy())
                    non0_cls_labels_val.extend(cls_y_binary_nn_val.cpu().numpy())
            
            # === "has0" 二分类 (来自专用的 cls_head_has0) ===
            # 这个头是在所有样本上评估的，使用 cls_y_has0_v_d 作为真实标签
            if pred_cls_binary_has0_dedicated_val_logits.size(0) > 0:
                 has0_cls_preds_val_dedicated.extend(pred_cls_binary_has0_dedicated_val_logits.argmax(dim=1).cpu().numpy())
                 has0_cls_labels_val_dedicated.extend(cls_y_has0_v_d.cpu().numpy()) # 使用 cls_y_has0_d
            # ===========================================

    # --- 计算所有指标 ---
    mae_val, corr_val = float('nan'), float('nan')
    acc5_val, acc7_val = float('nan'), float('nan')
    f1_non0_val, acc2_non0_val = float('nan'), float('nan')
    f1_has0_val_dedicated, acc2_has0_val_dedicated = float('nan'), float('nan') # _dedicated 后缀表示来自专用头
    
    num_total_samples_val = len(all_reg_labels_val) # Should be same as len(has0_cls_labels_val_dedicated)
    num_non0_samples_val = len(non0_cls_labels_val) if non0_cls_labels_val else 0


    if all_reg_labels_val:
        reg_preds_np_val = np.array(all_reg_preds_val).flatten()
        reg_labels_np_val = np.array(all_reg_labels_val).flatten()
        if len(reg_labels_np_val) > 0 : # Check if not empty
            mae_val = mean_absolute_error(reg_labels_np_val, reg_preds_np_val)
            if len(reg_preds_np_val) >= 2 and len(reg_labels_np_val) >=2 and \
               len(np.unique(reg_preds_np_val)) > 1 and len(np.unique(reg_labels_np_val)) > 1 :
                corr_val, _ = pearsonr(reg_preds_np_val, reg_labels_np_val)
                if np.isnan(corr_val): corr_val = 0.0
            else: corr_val = 0.0 # Default if pearsonr conditions not met
            
            acc7_clip_range = config.get("dataset_specifics", {}).get("acc7_clip_range", (-3.0, 3.0))
            acc7_val = multiclass_acc(reg_preds_np_val, reg_labels_np_val, clip_range=acc7_clip_range)
            acc5_clip_range = config.get("dataset_specifics", {}).get("acc5_clip_range", (-2.0, 2.0))
            acc5_val = multiclass_acc(reg_preds_np_val, reg_labels_np_val, clip_range=acc5_clip_range)

    if non0_cls_labels_val:
        non0_cls_preds_np_val = np.array(non0_cls_preds_val).flatten()
        non0_cls_labels_np_val = np.array(non0_cls_labels_val).flatten()
        num_non0_samples_val = len(non0_cls_labels_np_val) # Re-calculate here based on collected
        if num_non0_samples_val > 0:
            acc2_non0_val = sk_accuracy_score(non0_cls_labels_np_val, non0_cls_preds_np_val)
            f1_non0_val = f1_score(non0_cls_labels_np_val, non0_cls_preds_np_val, average="binary", pos_label=1, zero_division=0)

    # === 使用专用头的结果计算 'has0' 指标 ===
    if has0_cls_labels_val_dedicated:
        has0_cls_preds_np_val_dedicated = np.array(has0_cls_preds_val_dedicated).flatten()
        has0_cls_labels_np_val_dedicated = np.array(has0_cls_labels_val_dedicated).flatten()
        if len(has0_cls_labels_np_val_dedicated) > 0:
            acc2_has0_val_dedicated = sk_accuracy_score(has0_cls_labels_np_val_dedicated, has0_cls_preds_np_val_dedicated)
            f1_has0_val_dedicated = f1_score(has0_cls_labels_np_val_dedicated, has0_cls_preds_np_val_dedicated, average="binary", pos_label=1, zero_division=0)
    # ========================================

    w_f1_non0 = config["training"].get("score_weight_f1_non0", 0.4)
    w_f1_has0 = config["training"].get("score_weight_f1_has0", 0.1) # 这个权重现在对应 f1_has0_val_dedicated
    w_corr = config["training"].get("score_weight_corr", 0.3)
    w_mae = config["training"].get("score_weight_mae", 0.2)

    current_combined_score = (w_f1_non0 * (f1_non0_val if not np.isnan(f1_non0_val) else 0.0)) + \
                             (w_f1_has0 * (f1_has0_val_dedicated if not np.isnan(f1_has0_val_dedicated) else 0.0)) + \
                             (w_corr * (corr_val if not np.isnan(corr_val) else 0.0)) - \
                             (w_mae * (mae_val if not np.isnan(mae_val) else float('inf')))


    log_entry = ( f"Epoch {epoch+1}, TrainLoss: {avg_epoch_train_loss:.4f}\n"
                  f"  Val MAE: {mae_val:.4f}, Corr: {corr_val:.4f}\n"
                  f"  Val ACC-2 (non0): {acc2_non0_val:.4f}, F1 (non0): {f1_non0_val:.4f} (N={num_non0_samples_val})\n"
                  f"  Val ACC-2 (has0_dedicated): {acc2_has0_val_dedicated:.4f}, F1 (has0_dedicated): {f1_has0_val_dedicated:.4f} (N={len(has0_cls_labels_val_dedicated) if has0_cls_labels_val_dedicated else 0})\n"
                  f"  Val ACC-5: {acc5_val:.4f}, ACC-7: {acc7_val:.4f}\n"
                  f"  Val Combined Score: {current_combined_score:.4f}\n"
                )
    with open(log_file_main_train, "a", encoding="utf-8") as f_log:
        f_log.write(log_entry)
    print(log_entry) # Also print to console

    if current_combined_score > best_combined_score:
        best_combined_score = current_combined_score
        patience = 0
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model_comprehensive_has0head.pth")) # 更新模型文件名
        with open(log_file_main_train, "a", encoding="utf-8") as f_log:
            f_log.write(f"Saved best model (comprehensive metrics with has0_head) at epoch {epoch+1}\n")
    else:
        patience += 1
        if patience >= config["training"]["early_stopping_patience"]:
            with open(log_file_main_train, "a", encoding="utf-8") as f_log:
                f_log.write(f"Early stopping triggered at epoch {epoch+1}.\n")
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break