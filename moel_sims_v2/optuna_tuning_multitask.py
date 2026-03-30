import optuna
import torch
import yaml
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, accuracy_score as sk_accuracy_score, f1_score
from scipy.stats import pearsonr
from transformers import get_cosine_schedule_with_warmup
import pickle
import copy
import traceback
import random

# Seed, multiclass_acc, valid_seq_mask, clean_inf_nan, preprocess_data_for_optuna_comprehensive 保持不变
seed = 17
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def multiclass_acc(y_pred_reg, y_true_reg, clip_range=None):
    if clip_range:
        y_pred_reg = np.clip(y_pred_reg, clip_range[0], clip_range[1])
        y_true_reg = np.clip(y_true_reg, clip_range[0], clip_range[1])
    y_pred_class = np.round(y_pred_reg).astype(int)
    y_true_class = np.round(y_true_reg).astype(int)
    if len(y_true_class) == 0:
        return np.nan
    return sk_accuracy_score(y_true_class, y_pred_class)

def valid_seq_mask(x):
    return (x.abs().sum(dim=-1) > 0).sum(dim=1) >= 1

def clean_inf_nan(tensor):
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    return tensor

def preprocess_data_for_optuna_comprehensive(data_dict, split, 
                                             p_thresh_non_neutral, n_thresh_non_neutral, 
                                             p_thresh_has0, text_feature_key='text'): # Added text_feature_key
    text_data = data_dict[split].get(text_feature_key, data_dict[split].get('text', data_dict[split].get('words', data_dict[split].get('text_features'))))
    audio_data = data_dict[split].get('audio', data_dict[split].get('audio_features'))
    vision_data = data_dict[split].get('vision', data_dict[split].get('vision_features'))
    reg_labels_data = data_dict[split].get('regression_labels', data_dict[split].get('labels'))

    text = torch.tensor(text_data, dtype=torch.float32)
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
    
    text_final = text[base_valid_mask]
    audio_final = audio[base_valid_mask]
    vision_final = vision[base_valid_mask]
    reg_labels_final = original_reg_labels[base_valid_mask]

    if reg_labels_final.numel() == 0:
        print(f"警告: 在split '{split}' 中基础有效性过滤后没有样本。")
        # Use typical shapes or shapes from original data if possible
        text_shape_orig = text_data.shape if hasattr(text_data, 'shape') else (0, 300) # Default to a common dim
        audio_shape_orig = audio_data.shape if hasattr(audio_data, 'shape') else (0, 74)
        vision_shape_orig = vision_data.shape if hasattr(vision_data, 'shape') else (0, 35)

        return torch.empty(0, text_shape_orig[-1]), \
               torch.empty(0, audio_shape_orig[-1]), \
               torch.empty(0, vision_shape_orig[-1]), \
               torch.empty(0, 1), \
               torch.empty(0, dtype=torch.long), \
               torch.empty(0, dtype=torch.bool), \
               torch.empty(0, dtype=torch.long)

    reg_labels_squeezed = reg_labels_final.squeeze()

    is_positive_non0 = reg_labels_squeezed > p_thresh_non_neutral
    is_negative_non0 = reg_labels_squeezed < n_thresh_non_neutral
    non_neutral_mask_out = is_positive_non0 | is_negative_non0
    binary_cls_labels_non0 = torch.zeros(reg_labels_final.shape[0], dtype=torch.long)
    binary_cls_labels_non0[is_positive_non0] = 1

    binary_cls_labels_has0 = torch.zeros(reg_labels_final.shape[0], dtype=torch.long)
    binary_cls_labels_has0[reg_labels_squeezed > p_thresh_has0] = 1

    print(f"Split '{split}': Total valid samples {len(text_final)}, Non-neutral for 'non0' classification {non_neutral_mask_out.sum().item()}")
    return text_final, audio_final, vision_final, reg_labels_final, \
           binary_cls_labels_non0, non_neutral_mask_out, binary_cls_labels_has0


def objective(trial, base_config_dict, data_pickle_path, base_output_dir_for_study, dataset_name="sims-v2"):
    config = copy.deepcopy(base_config_dict)

    # --- 超参数建议 ---
    config["model"]["hidden_dim"] = trial.suggest_categorical("hidden_dim", [128, 256, 384, 512]) 
    config["model"]["dropout_prob"] = trial.suggest_float("dropout_prob", 0.1, 0.5)
    config["model"]["lambda_style"] = trial.suggest_float("lambda_style", 0.1, 1.0)
    # ... (其他 router, cls, reg 参数保持不变)
    config["model"]["router_hidden_dim"] = trial.suggest_categorical("router_hidden_dim", [64, 128, 256, 384])
    config["model"]["router_dropout"] = trial.suggest_float("router_dropout", 0.05, 0.4)
    config["model"]["cls_hidden_dim"] = trial.suggest_categorical("cls_hidden_dim", [64, 128, 256, 384])
    config["model"]["cls_dropout"] = trial.suggest_float("cls_dropout", 0.1, 0.6)
    config["model"]["reg_hidden_dim"] = trial.suggest_categorical("reg_hidden_dim", [64, 128, 256, 384]) 
    config["model"]["reg_dropout"] = trial.suggest_float("reg_dropout", 0.1, 0.6)


    config["training"]["lr"] = trial.suggest_float("lr", 1e-6, 1e-3, log=True) 
    config["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True) 
    config["training"]["loss_reg"] = trial.suggest_float("loss_reg", 0.5, 2.5) 
    config["training"]["loss_alpha"] = trial.suggest_float("loss_alpha", 0.5, 2.5) # For 'non0' head
    config["training"]["loss_beta"] = trial.suggest_float("loss_beta", 0.05, 1.0)   
    
    # === 新增：为 'has0' 专用头的损失权重建议 ===
    config["training"]["loss_weight_has0_cls"] = trial.suggest_float("loss_weight_has0_cls", 0.1, 2.0)
    # =========================================

    config["model"]["num_classes"] = 2 # Hardcoded for binary tasks
    focal_alpha_option = trial.suggest_categorical("focal_alpha_binary", ["balanced", "custom"])
    if focal_alpha_option == "balanced":
        config["training"]["focal_alpha"] = [0.5, 0.5] # For non0 head and aux heads
        config["training"]["focal_alpha_has0"] = [0.5, 0.5] # For dedicated has0 head
    else:
        alpha_class0_non0 = trial.suggest_float("focal_alpha_c0_non0", 0.25, 0.75)
        config["training"]["focal_alpha"] = [alpha_class0_non0, 1.0 - alpha_class0_non0]
        
        alpha_class0_has0 = trial.suggest_float("focal_alpha_c0_has0", 0.25, 0.75) # Potentially different for has0
        config["training"]["focal_alpha_has0"] = [alpha_class0_has0, 1.0 - alpha_class0_has0]

    config["training"]["focal_gamma"] = trial.suggest_float("focal_gamma", 1.0, 3.0) # For non0 and aux
    config["training"]["focal_gamma_has0"] = trial.suggest_float("focal_gamma_has0", 1.0, 3.0) # For dedicated has0


    config["training"]["aux_loss_weights"] = {
        "text_reg": trial.suggest_float("text_reg", 0.05, 0.8), # Renamed to avoid clash with main reg_loss if any
        "text_cls": trial.suggest_float("text_cls", 0.05, 0.8),
        "audio_reg": trial.suggest_float("audio_reg", 0.01, 0.5),
        "audio_cls": trial.suggest_float("audio_cls", 0.01, 0.5),
        "vision_reg": trial.suggest_float("vision_reg", 0.01, 0.5),
        "vision_cls": trial.suggest_float("vision_cls", 0.01, 0.5),
    }
    
    # Dataset specific dimensions (BERT: text_dim usually 768)
    if dataset_name == "mosi":
        config["model"]["text_dim"] = base_config_dict["model"].get("text_dim_mosi_bert", 768)
        config["model"]["audio_dim"] = base_config_dict["model"].get("audio_dim_mosi", 5) # Or appropriate for MOSI
        config["model"]["visual_dim"] = base_config_dict["model"].get("visual_dim_mosi", 20)
    elif dataset_name == "mosei":
        config["model"]["text_dim"] = base_config_dict["model"].get("text_dim_mosei_bert", 768)
        config["model"]["audio_dim"] = base_config_dict["model"].get("audio_dim_mosei", 74)
        config["model"]["visual_dim"] = base_config_dict["model"].get("visual_dim_mosei", 35)
    elif dataset_name == "sims-v2":
        config["model"]["text_dim"] = base_config_dict["model"].get("text_dim_sims-v2", 768)
        config["model"]["audio_dim"] = base_config_dict["model"].get("audio_dim_sims-v2", 25)
        config["model"]["visual_dim"] = base_config_dict["model"].get("visual_dim_sims-v2", 177)
    else:
        raise ValueError(f"未知的 dataset_name: {dataset_name}")
    
    trial_output_dir = os.path.join(base_output_dir_for_study, f"trial_{trial.number}")
    os.makedirs(trial_output_dir, exist_ok=True)
    config["paths"]["output_dir"] = trial_output_dir 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    POSITIVE_THRESHOLD_FOR_BINARY_NON_NEUTRAL = config.get("dataset_specifics", {}).get("positive_threshold_binary_non_neutral", 0.00001)
    NEGATIVE_THRESHOLD_FOR_BINARY_NON_NEUTRAL = config.get("dataset_specifics", {}).get("negative_threshold_binary_non_neutral", -0.00001)
    POSITIVE_THRESHOLD_FOR_BINARY_HAS0 = config.get("dataset_specifics", {}).get("positive_threshold_binary_has0", 0.0)
    
    text_feature_key_for_optuna = base_config_dict.get("dataset_specifics", {}).get(f"text_feature_key_{dataset_name}", 'text')


    with open(data_pickle_path, "rb") as f:
        loaded_data = pickle.load(f)
    
    train_t, train_a, train_v, train_reg_y, \
    train_cls_y_non0, train_non_neutral_mask, train_cls_y_has0 = preprocess_data_for_optuna_comprehensive(
        loaded_data, "train", 
        POSITIVE_THRESHOLD_FOR_BINARY_NON_NEUTRAL, NEGATIVE_THRESHOLD_FOR_BINARY_NON_NEUTRAL,
        POSITIVE_THRESHOLD_FOR_BINARY_HAS0, text_feature_key=text_feature_key_for_optuna
    )
    val_t, val_a, val_v, val_reg_y, \
    val_cls_y_non0, val_non_neutral_mask, val_cls_y_has0 = preprocess_data_for_optuna_comprehensive(
        loaded_data, "valid", 
        POSITIVE_THRESHOLD_FOR_BINARY_NON_NEUTRAL, NEGATIVE_THRESHOLD_FOR_BINARY_NON_NEUTRAL,
        POSITIVE_THRESHOLD_FOR_BINARY_HAS0, text_feature_key=text_feature_key_for_optuna
    )

    if train_t.numel() == 0 or val_t.numel() == 0:
        print(f"T{trial.number}: 训练集或验证集为空。剪枝。")
        # Save problematic config for debugging
        with open(os.path.join(trial_output_dir, f"optuna_pruned_empty_data_config_trial_{trial.number}.yaml"), "w") as f_cfg_err:
            yaml.dump(config, f_cfg_err)
        raise optuna.exceptions.TrialPruned("No data after preprocessing.")

    current_batch_size = config["training"].get("batch_size_mosi" if dataset_name == "mosi" else "batch_size", 32)
    train_loader = DataLoader(TensorDataset(train_t, train_a, train_v, train_reg_y, train_cls_y_non0, train_non_neutral_mask, train_cls_y_has0),
                              batch_size=current_batch_size, shuffle=True, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(TensorDataset(val_t, val_a, val_v, val_reg_y, val_cls_y_non0, val_non_neutral_mask, val_cls_y_has0),
                            batch_size=current_batch_size, pin_memory=True if device.type == 'cuda' else False)

    from style_moe_multitask_transformer_model import StyleMoEMultiTaskTransformer, FocalLoss # FocalLoss import is fine
    
    model = StyleMoEMultiTaskTransformer(config).to(device) # Model now has cls_head_has0
    criterion_reg = torch.nn.SmoothL1Loss()
    
    # Criterion for 'non0' and aux
    focal_alpha_tensor_non0_aux = torch.tensor(config["training"]["focal_alpha"], dtype=torch.float32).to(device)
    criterion_cls_non0_aux = FocalLoss(gamma=config["training"]["focal_gamma"], alpha=focal_alpha_tensor_non0_aux)
    criterion_cls_non0_aux.to(device)
    
    # Criterion for dedicated 'has0' head
    focal_alpha_tensor_has0 = torch.tensor(config["training"]["focal_alpha_has0"], dtype=torch.float32).to(device)
    criterion_cls_has0_dedicated = FocalLoss(gamma=config["training"]["focal_gamma_has0"], alpha=focal_alpha_tensor_has0)
    criterion_cls_has0_dedicated.to(device)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
    
    num_epochs_for_trial = config["training"].get("epochs_optuna_" + dataset_name, config["training"]["epochs"]) # Use specific optuna epochs
    num_total_steps = len(train_loader) * num_epochs_for_trial
    num_warmup_steps = int(num_total_steps * 0.1) if len(train_loader) > 5 else max(1, len(train_loader) // 2)
    if num_total_steps == 0 : num_total_steps = 1
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_total_steps)

    best_combined_score_for_trial = -float('inf')
    patience_counter = 0
    early_stopping_patience_for_trial = config["training"].get("patience_optuna_" + dataset_name, config["training"]["early_stopping_patience"])


    log_file_path = os.path.join(trial_output_dir, f"optuna_train_val_log_{dataset_name}_trial_{trial.number}.txt")
    with open(log_file_path, "a", encoding="utf-8") as f_log_init:
        f_log_init.write(f"--- Trial {trial.number} Config for {dataset_name.upper()} ---\n")
        yaml.dump(config, f_log_init, indent=2)
        f_log_init.write(f"--- End Trial Config ---\n")
        f_log_init.write(f"Train samples: {len(train_t)}, Valid samples: {len(val_t)}\n\n")
        
    aux_weights = config["training"].get("aux_loss_weights", {})
    loss_weight_has0_cls_val = config["training"]["loss_weight_has0_cls"] # Get tuned value

    for epoch in range(num_epochs_for_trial):
        model.train()
        epoch_train_loss_sum = 0.0
        num_train_batches = 0
        for t, a, v, reg_y_b, cls_y_non0_b, non_neutral_mask_b, cls_y_has0_b in train_loader:
            batch_mask_valid_input = (t.abs().sum(dim=(1,2)) + a.abs().sum(dim=(1,2)) + v.abs().sum(dim=(1,2))) > 0
            if not hasattr(t, 'shape') or t.shape[0] == 0 or batch_mask_valid_input.sum() == 0 : continue
            
            t_m,a_m,v_m,reg_y_m,cls_y_non0_m,non_neutral_m, cls_y_has0_m = \
                [x[batch_mask_valid_input] for x in [t,a,v,reg_y_b,cls_y_non0_b,non_neutral_mask_b, cls_y_has0_b]]
            if t_m.size(0) == 0: continue

            t_d,a_d,v_d,reg_y_d,cls_y_non0_d,non_neutral_d, cls_y_has0_d = \
                [x.to(device) for x in [t_m,a_m,v_m,reg_y_m,cls_y_non0_m,non_neutral_m, cls_y_has0_m]]

            outputs = model(t_d, a_d, v_d) # Returns 10 values now
            pred_reg, pred_cls_logits_non0, style_loss = outputs[0], outputs[1], outputs[2]
            aux_text_reg, aux_text_cls_logits = outputs[3], outputs[4]
            aux_audio_reg, aux_audio_cls_logits = outputs[5], outputs[6]
            aux_vision_reg, aux_vision_cls_logits = outputs[7], outputs[8]
            pred_cls_logits_has0_dedicated = outputs[9] # New output
            
            all_outputs_for_nan_check = [pred_reg, pred_cls_logits_non0, style_loss, 
                                     aux_text_reg, aux_text_cls_logits,
                                     aux_audio_reg, aux_audio_cls_logits,
                                     aux_vision_reg, aux_vision_cls_logits,
                                     pred_cls_logits_has0_dedicated] # Add new output to check
            if any(out is not None and torch.isnan(out).any() for out in all_outputs_for_nan_check):
                with open(log_file_path, "a", encoding="utf-8") as f_log_err:
                    f_log_err.write(f"T{trial.number} E{epoch+1}: NaN in model output. Skipping batch.\n")
                continue
            
            loss_r = criterion_reg(pred_reg, reg_y_d)
            loss_c_non0 = torch.tensor(0.0, device=device)
            if non_neutral_d.sum() > 0:
                pred_cls_logits_non0_filtered = pred_cls_logits_non0[non_neutral_d]
                cls_y_binary_d_non0_filtered = cls_y_non0_d[non_neutral_d]
                if pred_cls_logits_non0_filtered.size(0) > 0:
                    loss_c_non0 = criterion_cls_non0_aux(pred_cls_logits_non0_filtered, cls_y_binary_d_non0_filtered)

            # Loss for dedicated 'has0' head (on all samples)
            loss_c_has0_dedicated = torch.tensor(0.0, device=device)
            if pred_cls_logits_has0_dedicated.size(0) > 0:
                loss_c_has0_dedicated = criterion_cls_has0_dedicated(pred_cls_logits_has0_dedicated, cls_y_has0_d)

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
            
            all_losses_for_nan_check = [loss_r, loss_c_non0, style_loss, loss_c_has0_dedicated, # Add new loss
                                    loss_r_text_aux, loss_c_text_aux, 
                                    loss_r_audio_aux, loss_c_audio_aux,
                                    loss_r_vision_aux, loss_c_vision_aux]
            if any(l is not None and torch.isnan(l).any() for l in all_losses_for_nan_check):
                with open(log_file_path, "a", encoding="utf-8") as f_log_err:
                    f_log_err.write(f"T{trial.number} E{epoch+1}: NaN in loss components. Skipping batch.\n")
                continue

            total_loss = (config["training"]["loss_reg"] * loss_r + 
                          config["training"]["loss_alpha"] * loss_c_non0 + 
                          config["training"]["loss_beta"] * style_loss +
                          loss_weight_has0_cls_val * loss_c_has0_dedicated) # Add weighted new loss
            
            total_loss += aux_weights.get("text_reg", 0.0) * loss_r_text_aux # Use renamed aux weight keys
            total_loss += aux_weights.get("text_cls", 0.0) * loss_c_text_aux
            total_loss += aux_weights.get("audio_reg", 0.0) * loss_r_audio_aux
            total_loss += aux_weights.get("audio_cls", 0.0) * loss_c_audio_aux
            total_loss += aux_weights.get("vision_reg", 0.0) * loss_r_vision_aux
            total_loss += aux_weights.get("vision_cls", 0.0) * loss_c_vision_aux
            
            if torch.isnan(total_loss).any():
                with open(log_file_path, "a", encoding="utf-8") as f_log_err:
                     f_log_err.write(f"T{trial.number} E{epoch+1}: NaN in total loss. Skipping batch.\n")
                continue

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_train_loss_sum += total_loss.item()
            num_train_batches +=1
        
        avg_train_loss = epoch_train_loss_sum / num_train_batches if num_train_batches > 0 else float('nan')

        # --- 验证阶段 ---
        model.eval()
        all_reg_preds_val, all_reg_labels_val = [], []
        non0_cls_preds_val, non0_cls_labels_val = [], []
        has0_cls_preds_val_dedicated, has0_cls_labels_val_dedicated = [], [] # For dedicated head
        
        with torch.no_grad():
            for t_val, a_val, v_val, reg_y_val_b, cls_y_non0_val_b, non_neutral_mask_val_b, cls_y_has0_val_b in val_loader:
                if not hasattr(t_val, 'shape') or t_val.shape[0] == 0: continue
                
                t_v_d,a_v_d,v_v_d,reg_y_v_d,cls_y_non0_v_d,non_neutral_v_d, cls_y_has0_v_d = \
                    [x.to(device) for x in [t_val,a_val,v_val,reg_y_val_b,cls_y_non0_val_b,non_neutral_mask_val_b, cls_y_has0_val_b]]

                outputs_val = model(t_v_d, a_v_d, v_v_d)
                pred_reg_val = outputs_val[0]
                pred_cls_binary_non0_val_logits = outputs_val[1]
                pred_cls_binary_has0_dedicated_val_logits = outputs_val[9] # New output
                
                if torch.isnan(pred_reg_val).any() or \
                   torch.isnan(pred_cls_binary_non0_val_logits).any() or \
                   torch.isnan(pred_cls_binary_has0_dedicated_val_logits).any():
                    with open(log_file_path, "a", encoding="utf-8") as f_log_err:
                         f_log_err.write(f"T{trial.number} E{epoch+1} VAL: NaN in model output. Skipping batch.\n")
                    continue
                    
                all_reg_preds_val.extend(pred_reg_val.cpu().numpy())
                all_reg_labels_val.extend(reg_y_v_d.cpu().numpy())

                if non_neutral_v_d.sum() > 0:
                    pred_cls_logits_nn_val = pred_cls_binary_non0_val_logits[non_neutral_v_d]
                    cls_y_binary_nn_val = cls_y_non0_v_d[non_neutral_v_d]
                    if pred_cls_logits_nn_val.size(0) > 0 :
                        non0_cls_preds_val.extend(pred_cls_logits_nn_val.argmax(dim=1).cpu().numpy())
                        non0_cls_labels_val.extend(cls_y_binary_nn_val.cpu().numpy())
                
                # Use dedicated head for 'has0' metrics
                if pred_cls_binary_has0_dedicated_val_logits.size(0) > 0:
                    has0_cls_preds_val_dedicated.extend(pred_cls_binary_has0_dedicated_val_logits.argmax(dim=1).cpu().numpy())
                    has0_cls_labels_val_dedicated.extend(cls_y_has0_v_d.cpu().numpy())
        
        mae_val, corr_val = float('nan'), float('nan')
        acc5_val, acc7_val = float('nan'), float('nan')
        f1_non0_val, acc2_non0_val = float('nan'), float('nan')
        f1_has0_val_dedicated, acc2_has0_val_dedicated = float('nan'), float('nan') # Renamed
        
        num_total_samples_val = len(all_reg_labels_val)
        num_non0_samples_val = len(non0_cls_labels_val) if non0_cls_labels_val else 0
        num_has0_samples_val_dedicated = len(has0_cls_labels_val_dedicated) if has0_cls_labels_val_dedicated else 0


        if all_reg_labels_val:
            reg_preds_np_val = np.array(all_reg_preds_val).flatten()
            reg_labels_np_val = np.array(all_reg_labels_val).flatten()
            if len(reg_labels_np_val) > 0: # Check non-empty
                mae_val = mean_absolute_error(reg_labels_np_val, reg_preds_np_val)
                if len(reg_preds_np_val) >= 2 and len(reg_labels_np_val) >=2 and \
                   len(np.unique(reg_preds_np_val)) > 1 and len(np.unique(reg_labels_np_val)) > 1 :
                    corr_val, _ = pearsonr(reg_preds_np_val, reg_labels_np_val)
                    if np.isnan(corr_val): corr_val = 0.0
                else: corr_val = 0.0
                
                acc7_clip_range = config.get("dataset_specifics", {}).get("acc7_clip_range", (-3.0, 3.0))
                acc7_val = multiclass_acc(reg_preds_np_val, reg_labels_np_val, clip_range=acc7_clip_range)
                acc5_clip_range = config.get("dataset_specifics", {}).get("acc5_clip_range", (-2.0, 2.0))
                acc5_val = multiclass_acc(reg_preds_np_val, reg_labels_np_val, clip_range=acc5_clip_range)

        if non0_cls_labels_val: # Check if list is populated
            non0_cls_preds_np_val = np.array(non0_cls_preds_val).flatten()
            non0_cls_labels_np_val = np.array(non0_cls_labels_val).flatten()
            num_non0_samples_val = len(non0_cls_labels_np_val)
            if num_non0_samples_val > 0:
                acc2_non0_val = sk_accuracy_score(non0_cls_labels_np_val, non0_cls_preds_np_val)
                f1_non0_val = f1_score(non0_cls_labels_np_val, non0_cls_preds_np_val, average="binary", pos_label=1, zero_division=0)

        if has0_cls_labels_val_dedicated: # Check if list is populated
            has0_cls_preds_np_val_dedicated = np.array(has0_cls_preds_val_dedicated).flatten()
            has0_cls_labels_np_val_dedicated = np.array(has0_cls_labels_val_dedicated).flatten()
            num_has0_samples_val_dedicated = len(has0_cls_labels_np_val_dedicated)
            if num_has0_samples_val_dedicated > 0:
                acc2_has0_val_dedicated = sk_accuracy_score(has0_cls_labels_np_val_dedicated, has0_cls_preds_np_val_dedicated)
                f1_has0_val_dedicated = f1_score(has0_cls_labels_np_val_dedicated, has0_cls_preds_np_val_dedicated, average="binary", pos_label=1, zero_division=0)
        
        w_f1_non0 = config["training"].get("score_weight_f1_non0", 0.4)
        w_f1_has0 = config["training"].get("score_weight_f1_has0", 0.1) 
        w_corr = config["training"].get("score_weight_corr", 0.3)
        w_mae = config["training"].get("score_weight_mae", 0.2) 
        w_acc5 = config["training"].get("score_weight_acc5", 0.0)
        w_acc7 = config["training"].get("score_weight_acc7", 0.0)

        current_epoch_combined_score = (
            w_f1_non0 * (f1_non0_val if not np.isnan(f1_non0_val) else 0.0) +
            w_f1_has0 * (f1_has0_val_dedicated if not np.isnan(f1_has0_val_dedicated) else 0.0) + # Use dedicated F1
            w_corr    * (corr_val if not np.isnan(corr_val) else 0.0) -
            w_mae     * (mae_val if not np.isnan(mae_val) else float('inf')) +
            w_acc5    * (acc5_val if not np.isnan(acc5_val) else 0.0) +
            w_acc7    * (acc7_val if not np.isnan(acc7_val) else 0.0)
        )
        
        with open(log_file_path, "a", encoding="utf-8") as f:
            log_score = current_epoch_combined_score if not (np.isinf(current_epoch_combined_score) or np.isnan(current_epoch_combined_score)) else float('nan')
            f.write(f"Epoch {epoch+1}/{num_epochs_for_trial}, TrainLoss: {avg_train_loss:.4f}, ValidCombined: {log_score:.4f}\n")
            f.write(f"  MAE: {mae_val:.4f}, Corr: {corr_val:.4f}\n")
            f.write(f"  ACC-2 (non0): {acc2_non0_val:.4f}, F1 (non0): {f1_non0_val:.4f} (N={num_non0_samples_val})\n")
            f.write(f"  ACC-2 (has0_dedicated): {acc2_has0_val_dedicated:.4f}, F1 (has0_dedicated): {f1_has0_val_dedicated:.4f} (N={num_has0_samples_val_dedicated})\n") # Updated
            f.write(f"  ACC-5: {acc5_val:.4f}, ACC-7: {acc7_val:.4f}\n")

        if current_epoch_combined_score > best_combined_score_for_trial:
            best_combined_score_for_trial = current_epoch_combined_score
            patience_counter = 0
            # Optionally save best model per trial if needed for later inspection
            # torch.save(model.state_dict(), os.path.join(trial_output_dir, f"best_model_trial_{trial.number}.pth"))
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience_for_trial:
            print(f"T{trial.number}: Early stopping at epoch {epoch+1}.")
            with open(log_file_path, "a", encoding="utf-8") as f_log_earlystop:
                 f_log_earlystop.write(f"T{trial.number}: Early stopping at epoch {epoch+1}.\n")
            break
        
        # Report a valid score, ensuring it's not NaN or Inf
        report_value = -1e9 # Default for invalid scores
        if best_combined_score_for_trial is not None and \
           not np.isnan(best_combined_score_for_trial) and \
           not np.isinf(best_combined_score_for_trial):
            report_value = best_combined_score_for_trial
        
        trial.report(report_value, epoch) 
        if trial.should_prune():
            print(f"T{trial.number}: Pruning at epoch {epoch+1}.")
            with open(log_file_path, "a", encoding="utf-8") as f_log_prune:
                 f_log_prune.write(f"T{trial.number}: Pruning at epoch {epoch+1}.\n")
            # Save problematic config for debugging
            with open(os.path.join(trial_output_dir, f"optuna_pruned_config_trial_{trial.number}.yaml"), "w") as f_cfg_err:
                 yaml.dump(config, f_cfg_err)
            raise optuna.exceptions.TrialPruned()
            
    final_score_to_return = -1e9
    if best_combined_score_for_trial is not None and \
       not np.isnan(best_combined_score_for_trial) and \
       not np.isinf(best_combined_score_for_trial) and \
       best_combined_score_for_trial != -float('inf'):
        final_score_to_return = best_combined_score_for_trial
    else:
        print(f"T{trial.number} failed to produce a valid combined score ({best_combined_score_for_trial}). Returning very low score.")
        with open(log_file_path, "a", encoding="utf-8") as f_log_fail:
            f_log_fail.write(f"T{trial.number} failed to produce a valid combined score ({best_combined_score_for_trial}).\n")
        # Save problematic config for debugging
        with open(os.path.join(trial_output_dir, f"optuna_failed_score_config_trial_{trial.number}.yaml"), "w") as f_cfg_err:
            yaml.dump(config, f_cfg_err)
            
    return final_score_to_return


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_name = "config.yaml" 
    config_path = os.path.join(script_dir, config_file_name)

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    
    base_config["model"]["num_classes"] = 2
    if "aux_loss_weights" not in base_config["training"]:
         base_config["training"]["aux_loss_weights"] = {} 
    if "focal_alpha" not in base_config["training"] or len(base_config["training"]["focal_alpha"]) != 2:
        base_config["training"]["focal_alpha"] = [0.5, 0.5]
    # Ensure default for has0 alpha/gamma if not present (will be overridden by Optuna if in search space)
    if "focal_alpha_has0" not in base_config["training"]:
        base_config["training"]["focal_alpha_has0"] = base_config["training"]["focal_alpha"]
    if "focal_gamma_has0" not in base_config["training"]:
        base_config["training"]["focal_gamma_has0"] = base_config["training"].get("focal_gamma", 2.0)


    DATASET_BEING_TUNED = base_config.get("dataset_name_for_tuning", "sims-v2") 
    
    base_config.setdefault("dataset_specifics", {})
    # Common BERT feature key, can be overridden by dataset_specifics in config
    text_feature_key = base_config.get("dataset_specifics",{}).get(f"text_feature_key_{DATASET_BEING_TUNED}", 'text')
    base_config["dataset_specifics"][f"text_feature_key_{DATASET_BEING_TUNED}"] = text_feature_key # Ensure it's set for optuna

    if DATASET_BEING_TUNED == "mosei":
        base_config["dataset_specifics"]["positive_threshold_binary_non_neutral"] = 0.00001
        base_config["dataset_specifics"]["negative_threshold_binary_non_neutral"] = -0.00001
        base_config["dataset_specifics"]["positive_threshold_binary_has0"] = 0.0
        base_config["dataset_specifics"]["acc7_clip_range"] = (-3.0, 3.0)
        base_config["dataset_specifics"]["acc5_clip_range"] = (-2.0, 2.0)
        base_config["training"]["epochs_optuna_mosei"] = base_config["training"].get("epochs_optuna_mosei", 50) 
        base_config["training"]["patience_optuna_mosei"] = base_config["training"].get("patience_optuna_mosei", 10)
        data_pickle_filename = base_config.get("data_filename_mosei_bert", "aligned_50.pkl")

    elif DATASET_BEING_TUNED == "mosi":
        base_config["dataset_specifics"]["positive_threshold_binary_non_neutral"] = 0.00001
        base_config["dataset_specifics"]["negative_threshold_binary_non_neutral"] = -0.00001
        base_config["dataset_specifics"]["positive_threshold_binary_has0"] = 0.0
        base_config["dataset_specifics"]["acc7_clip_range"] = (-3.0, 3.0) 
        base_config["dataset_specifics"]["acc5_clip_range"] = (-2.0, 2.0)
        base_config["training"]["epochs_optuna_mosi"] = base_config["training"].get("epochs_optuna_mosi", 25)
        base_config["training"]["patience_optuna_mosi"] = base_config["training"].get("patience_optuna_mosi", 7)
        data_pickle_filename = base_config.get("data_filename_mosi_bert", "aligned_50.pkl") 
    elif DATASET_BEING_TUNED == "sims-v2":
        base_config["dataset_specifics"]["positive_threshold_binary_non_neutral"] = 0.00001
        base_config["dataset_specifics"]["negative_threshold_binary_non_neutral"] = -0.00001
        base_config["dataset_specifics"]["positive_threshold_binary_has0"] = 0.0
        base_config["dataset_specifics"]["acc7_clip_range"] = (-3.0, 3.0) 
        base_config["dataset_specifics"]["acc5_clip_range"] = (-2.0, 2.0)
        base_config["training"]["epochs_optuna_sims-v2"] = base_config["training"].get("epochs_optuna_sims-v2", 50)
        base_config["training"]["patience_optuna_sims-v2"] = base_config["training"].get("patience_optuna_sims-v2", 10)
        data_pickle_filename = base_config.get("data_filename_mosi_bert", "unaligned.pkl") 
    else:
        print(f"错误: 未知的 DATASET_BEING_TUNED: {DATASET_BEING_TUNED}")
        exit(1)

    data_dir_from_config = base_config["paths"]["data_dir"]
    data_pickle_path_for_tuning = os.path.join(data_dir_from_config, data_pickle_filename)
    if not os.path.exists(data_pickle_path_for_tuning):
        print(f"错误: Optuna数据文件 {data_pickle_path_for_tuning} 未找到！")
        exit(1)
    
    study_base_output_dir = os.path.join(base_config["paths"]["output_dir"], f"optuna_study_{DATASET_BEING_TUNED}_has0head_v1") # Update study name
    os.makedirs(study_base_output_dir, exist_ok=True)

    try:
        from style_moe_multitask_transformer_model import StyleMoEMultiTaskTransformer, FocalLoss
        print("StyleMoEMultiTaskTransformer and FocalLoss imported successfully for Optuna.")
    except ImportError as e:
        print(f"错误: 无法导入 StyleMoEMultiTaskTransformer 或 FocalLoss for Optuna: {e}")
        traceback.print_exc()
        exit(1)
    
    study_name = f"{DATASET_BEING_TUNED}_has0head_v1" # Update study name
    # Use a unique database file per study to avoid conflicts if Optuna is run for different major model versions
    storage_path = f"sqlite:///{os.path.join(study_base_output_dir, study_name + '.db')}"
    
    # Ensure epochs for pruner warmup steps is correctly derived
    epochs_for_pruner = base_config["training"].get(f"epochs_optuna_{DATASET_BEING_TUNED}", base_config["training"]["epochs"])
    pruner_warmup_steps = max(5, epochs_for_pruner // 4)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=base_config.get("optuna_startup_trials", 5), 
        n_warmup_steps=pruner_warmup_steps, 
        interval_steps=1 
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True, # Set to False if you want a fresh study for this version
        direction="maximize",
        pruner=pruner
    )

    objective_with_fixed_args = lambda trial: objective(
        trial,
        base_config, 
        data_pickle_path_for_tuning,
        study_base_output_dir,
        dataset_name=DATASET_BEING_TUNED
    )

    num_trials = base_config.get("optuna_num_trials", 100)
    timeout_seconds = base_config.get("optuna_timeout_seconds", 3600 * 24) # e.g., 24 hours

    try:
        study.optimize(objective_with_fixed_args, n_trials=num_trials, timeout=timeout_seconds, gc_after_trial=True) # Added gc_after_trial
    except KeyboardInterrupt:
        print("Optuna study interrupted by user.")
    except Exception as e:
        print(f"An error occurred during Optuna optimization: {e}")
        traceback.print_exc()
    finally:
        print("\nOptuna 研究统计: ")
        print(f"  已完成试验次数: {len(study.trials)}")
        
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and not np.isnan(t.value) and not np.isinf(t.value)]
        
        if completed_trials:
            best_trial_overall = None
            try: # Optuna's study.best_trial can raise ValueError if no completed trials
                if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
                    best_trial_overall = study.best_trial
                    # Validate if best_trial actually has a valid score
                    if best_trial_overall and (best_trial_overall.value is None or np.isnan(best_trial_overall.value) or np.isinf(best_trial_overall.value)):
                        # Manually find best if study.best_trial is problematic or returned a failed/pruned trial
                        best_value_manual = -float('inf')
                        best_trial_manual_candidate = None
                        for t_iter in completed_trials: # Iterate only through successfully completed trials
                            if t_iter.value > best_value_manual : 
                                best_value_manual = t_iter.value
                                best_trial_manual_candidate = t_iter
                        best_trial_overall = best_trial_manual_candidate
            except ValueError: 
                 best_trial_overall = None # No trials completed or other issue
            except Exception as e_study:
                 print(f"Error accessing study.best_trial, attempting manual selection: {e_study}")
                 best_trial_overall = None 
                 if completed_trials: 
                    best_value_manual = -float('inf')
                    for t_iter in completed_trials:
                        if t_iter.value is not None and not np.isnan(t_iter.value) and not np.isinf(t_iter.value):
                           if t_iter.value > best_value_manual:
                                best_value_manual = t_iter.value
                                best_trial_overall = t_iter


            if best_trial_overall and best_trial_overall.value is not None and not np.isnan(best_trial_overall.value) and not np.isinf(best_trial_overall.value):
                print("\n最佳试验详情:")
                print(f"  试验编号: {best_trial_overall.number}")
                print(f"  目标值 (Maximized Combined Score): {best_trial_overall.value:.4f}")
                print("  最佳超参数组合: ")
                for key, value in best_trial_overall.params.items():
                    print(f"    {key}: {value}")
                # Save best params to a yaml file
                best_params_file = os.path.join(study_base_output_dir, f"best_params_{study_name}.yaml")
                with open(best_params_file, "w") as f_best:
                    yaml.dump(best_trial_overall.params, f_best, indent=2)
                print(f"最佳参数已保存到: {best_params_file}")
            else:
                print("\n未能找到有效的最佳试验。")
        else:
            print("\n没有成功完成的试验。")
        print(f"\nOptuna研究 '{study_name}' 数据保存在: {storage_path}")
        print(f"每个试验的详细日志和模型（如果保存）位于: {study_base_output_dir}/trial_<number>/")