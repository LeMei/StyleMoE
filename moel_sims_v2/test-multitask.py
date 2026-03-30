# test-multitask.py (修改后版本)

import torch
import yaml
import pickle
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, accuracy_score as sk_accuracy_score, f1_score
from scipy.stats import pearsonr
from style_moe_multitask_transformer_model import StyleMoEMultiTaskTransformer
import argparse # [MODIFIED] 1. 导入 argparse

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

# === 配置读取 (修改后) ===
# [MODIFIED] 2. 添加命令行参数解析器
parser = argparse.ArgumentParser(description="Test a multi-task model.")
parser.add_argument('--config_path', type=str, required=True, 
                    help='Path to the specific run configuration file created by the pipeline.')
args = parser.parse_args()

# [MODIFIED] 3. 直接从指定的路径加载配置文件
with open(args.config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
print(f"成功加载用于测试的配置文件: {args.config_path}")

# [MODIFIED] 4. 从加载的 config 中直接获取实验名称和路径
# `experiment_name` 现在由主控脚本在运行时写入到专用的配置文件中
experiment_name_to_test = config.get("experiment_name", "ch-sims-v2_multiloss_comprehensive_metrics_has0head")
model_save_parent_dir = config["paths"]["output_dir"]
model_save_directory = os.path.join(model_save_parent_dir, experiment_name_to_test)


# --- 后续代码基本保持不变，因为它们都依赖于`config`字典 ---

if config["model"].get("num_classes") != 2:
    print(f"测试警告: 配置文件 num_classes ({config['model'].get('num_classes')}) 不是2。强制设为2。")
    config["model"]["num_classes"] = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = config["paths"]["data_dir"]

model_filename = config.get("testing", {}).get("model_filename", "best_model_comprehensive_has0head.pth")
model_path = os.path.join(model_save_directory, model_filename)

POSITIVE_THRESHOLD_FOR_BINARY_NON_NEUTRAL = config.get("dataset_specifics", {}).get("positive_threshold_binary_non_neutral", 0.00001) 
NEGATIVE_THRESHOLD_FOR_BINARY_NON_NEUTRAL = config.get("dataset_specifics", {}).get("negative_threshold_binary_non_neutral", -0.00001)
POSITIVE_THRESHOLD_FOR_BINARY_HAS0 = config.get("dataset_specifics", {}).get("positive_threshold_binary_has0", 0.0) 

dataset_name = os.path.basename(data_dir) 
data_file_path = os.path.join(data_dir, f"unaligned.pkl")
if not os.path.exists(data_file_path):
    data_file_path_alt = os.path.join(data_dir, "unaligned.pkl")
    if os.path.exists(data_file_path_alt):
        data_file_path = data_file_path_alt
    else:
        print(f"错误: 测试数据文件 {data_file_path} 或备选路径均未找到!")
        exit(1)
print(f"加载测试数据从: {data_file_path}")
with open(data_file_path, "rb") as f:
    data_loaded_full = pickle.load(f)

# === 数据预处理函数 (valid_seq_mask, clean_inf_nan, preprocess_test - 保持不变) ===
def valid_seq_mask(x):
    return (x.abs().sum(dim=-1) > 0).sum(dim=1) >= 1

def clean_inf_nan(tensor):
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    return tensor

def preprocess_test(split_data_dict, p_thresh_non_neutral, n_thresh_non_neutral, p_thresh_has0, text_feature_key='text'):
    text_data = split_data_dict.get(text_feature_key, split_data_dict.get('text', split_data_dict.get('words', split_data_dict.get('text_features'))))
    audio_data = split_data_dict.get('audio', split_data_dict.get('audio_features'))
    vision_data = split_data_dict.get('vision', split_data_dict.get('vision_features'))
    reg_labels_data = split_data_dict.get('regression_labels', split_data_dict.get('labels'))

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
    
    text_out = text[base_valid_mask]
    audio_out = audio[base_valid_mask]
    vision_out = vision[base_valid_mask]
    reg_labels_out = original_reg_labels[base_valid_mask]

    if reg_labels_out.numel() == 0:
        print(f"警告: 测试集基础过滤后无样本。")
        text_shape_orig = text_data.shape if hasattr(text_data, 'shape') else (0, config["model"]["text_dim"])
        audio_shape_orig = audio_data.shape if hasattr(audio_data, 'shape') else (0, config["model"]["audio_dim"])
        vision_shape_orig = vision_data.shape if hasattr(vision_data, 'shape') else (0, config["model"]["visual_dim"])
        return torch.empty(0, text_shape_orig[-1]), \
               torch.empty(0, audio_shape_orig[-1]), \
               torch.empty(0, vision_shape_orig[-1]), \
               torch.empty(0,1), \
               torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.bool), \
               torch.empty(0, dtype=torch.long)

    reg_labels_squeezed = reg_labels_out.squeeze()
    
    is_positive_non_neutral = reg_labels_squeezed > p_thresh_non_neutral
    is_negative_non_neutral = reg_labels_squeezed < n_thresh_non_neutral
    non_neutral_mask_out = is_positive_non_neutral | is_negative_non_neutral
    binary_cls_labels_non0_out = torch.zeros(reg_labels_out.shape[0], dtype=torch.long)
    binary_cls_labels_non0_out[is_positive_non_neutral] = 1

    binary_cls_labels_has0_out = torch.zeros(reg_labels_out.shape[0], dtype=torch.long)
    binary_cls_labels_has0_out[reg_labels_squeezed > p_thresh_has0] = 1

    print(f"Test Split: Total valid samples {len(text_out)}, Non-neutral for 'non0' classification {non_neutral_mask_out.sum().item()}")
    return text_out, audio_out, vision_out, reg_labels_out, binary_cls_labels_non0_out, non_neutral_mask_out, binary_cls_labels_has0_out

if 'test' not in data_loaded_full:
    print(f"错误: 加载的数据中未找到 'test' split。可用的 splits: {list(data_loaded_full.keys())}")
    exit(1)

test_text, test_audio, test_vision, test_reg_labels, \
test_cls_labels_non0, test_non_neutral_mask, test_cls_labels_has0 = preprocess_test(
    data_loaded_full["test"], 
    POSITIVE_THRESHOLD_FOR_BINARY_NON_NEUTRAL, 
    NEGATIVE_THRESHOLD_FOR_BINARY_NON_NEUTRAL,
    POSITIVE_THRESHOLD_FOR_BINARY_HAS0,
    text_feature_key=config.get("dataset_specifics",{}).get("text_feature_key_test", 'text')
)

if test_text.numel() == 0:
    print("错误: 测试集预处理后没有有效样本。")
    exit()

test_loader = DataLoader(
    TensorDataset(test_text, test_audio, test_vision, test_reg_labels, 
                  test_cls_labels_non0, test_non_neutral_mask, test_cls_labels_has0),
    batch_size=config["training"].get("batch_size_test", config["training"]["batch_size"])
)

model = StyleMoEMultiTaskTransformer(config).to(device)
if not os.path.exists(model_path):
    print(f"错误: 模型文件 {model_path} 未找到!")
    exit(1)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except RuntimeError as e:
    print(f"加载模型权重时出错: {e}")
    print("这可能是因为模型结构已更改，但正在加载旧的权重文件。请确保权重文件与当前模型结构匹配。")
    exit(1)
model.eval()

# --- 模型评估循环 (保持不变) ---
all_reg_preds_list, all_reg_labels_list = [], []
non0_cls_preds_list, non0_cls_labels_list = [], []
has0_cls_preds_list_dedicated, has0_cls_labels_list_dedicated = [], []

with torch.no_grad():
    for t, a, v, reg_y_b, cls_y_non0_b, non_neutral_mask_b, cls_y_has0_b in test_loader:
        if not hasattr(t, 'shape') or t.shape[0] == 0: continue
        
        t_d,a_d,v_d,reg_y_d,cls_y_non0_d,non_neutral_d, cls_y_has0_d = \
            [x.to(device) for x in [t,a,v,reg_y_b,cls_y_non0_b,non_neutral_mask_b, cls_y_has0_b]]
        
        outputs_batch = model(t_d, a_d, v_d)
        pred_reg_batch = outputs_batch[0]
        pred_cls_binary_non0_batch_logits = outputs_batch[1]
        pred_cls_binary_has0_dedicated_batch_logits = outputs_batch[9]
        
        all_reg_preds_list.extend(pred_reg_batch.cpu().numpy())
        all_reg_labels_list.extend(reg_y_d.cpu().numpy())
        
        if non_neutral_d.sum() > 0:
            pred_cls_logits_nn = pred_cls_binary_non0_batch_logits[non_neutral_d]
            cls_y_binary_nn = cls_y_non0_d[non_neutral_d]
            if pred_cls_logits_nn.size(0) > 0 :
                non0_cls_preds_list.extend(pred_cls_logits_nn.argmax(dim=1).cpu().numpy())
                non0_cls_labels_list.extend(cls_y_binary_nn.cpu().numpy())
        
        if pred_cls_binary_has0_dedicated_batch_logits.size(0) > 0:
            has0_cls_preds_list_dedicated.extend(pred_cls_binary_has0_dedicated_batch_logits.argmax(dim=1).cpu().numpy())
            has0_cls_labels_list_dedicated.extend(cls_y_has0_d.cpu().numpy())

# --- 指标计算 (保持不变) ---
all_reg_preds_np = np.array(all_reg_preds_list).flatten()
all_reg_labels_np = np.array(all_reg_labels_list).flatten()
non0_cls_preds_np = np.array(non0_cls_preds_list).flatten() if non0_cls_preds_list else np.array([])
non0_cls_labels_np = np.array(non0_cls_labels_list).flatten() if non0_cls_labels_list else np.array([])
has0_cls_preds_np_dedicated = np.array(has0_cls_preds_list_dedicated).flatten() if has0_cls_preds_list_dedicated else np.array([])
has0_cls_labels_np_dedicated = np.array(has0_cls_labels_list_dedicated).flatten() if has0_cls_labels_list_dedicated else np.array([])

mae, corr = float('nan'), float('nan')
acc5, acc7 = float('nan'), float('nan')
f1_non0, acc2_non0 = float('nan'), float('nan')
f1_has0_dedicated, acc2_has0_dedicated = float('nan'), float('nan')

num_total_test_samples = len(all_reg_labels_np)
num_non0_samples_test = len(non0_cls_labels_np)
num_has0_samples_test_dedicated = len(has0_cls_labels_np_dedicated)

if num_total_test_samples > 0:
    mae = mean_absolute_error(all_reg_labels_np, all_reg_preds_np)
    if len(all_reg_preds_np) >= 2 and len(all_reg_labels_np) >=2 and \
       len(np.unique(all_reg_preds_np)) > 1 and len(np.unique(all_reg_labels_np)) > 1 :
        corr, _ = pearsonr(all_reg_preds_np, all_reg_labels_np)
        if np.isnan(corr): corr = 0.0
    else: corr = 0.0
    
    acc7_clip_range = config.get("dataset_specifics", {}).get("acc7_clip_range", (-3.0, 3.0))
    acc7 = multiclass_acc(all_reg_preds_np, all_reg_labels_np, clip_range=acc7_clip_range)
    acc5_clip_range = config.get("dataset_specifics", {}).get("acc5_clip_range", (-2.0, 2.0))
    acc5 = multiclass_acc(all_reg_preds_np, all_reg_labels_np, clip_range=acc5_clip_range)
else:
    print("警告: 测试集回归样本不足。")

if num_non0_samples_test > 0:
    acc2_non0 = sk_accuracy_score(non0_cls_labels_np, non0_cls_preds_np)
    f1_non0 = f1_score(non0_cls_labels_np, non0_cls_preds_np, average="binary", pos_label=1, zero_division=0)
else:
    print("警告: 测试集'non0'二分类样本不足。")
    
if num_has0_samples_test_dedicated > 0:
    acc2_has0_dedicated = sk_accuracy_score(has0_cls_labels_np_dedicated, has0_cls_preds_np_dedicated)
    f1_has0_dedicated = f1_score(has0_cls_labels_np_dedicated, has0_cls_preds_np_dedicated, average="binary", pos_label=1, zero_division=0)
else:
    print("警告: 测试集'has0_dedicated'二分类样本不足。")

# --- 结果输出 (保持不变) ---
results_output_filename = config.get("testing", {}).get("results_filename", "test_results_comprehensive_has0head.txt")
results_output_path = os.path.join(model_save_directory, results_output_filename)

with open(results_output_path, "w", encoding="utf-8") as f:
    f.write(f"Test Set Performance (Model: {model_filename})\n")
    f.write(f"Dataset: {dataset_name.upper()}\n")
    f.write(f"Total valid test samples processed: {num_total_test_samples}\n")
    f.write(f"Model loaded from: {model_path}\n")
    # 使用 args.config_path 记录是哪个配置文件生成的此结果
    f.write(f"Config loaded from: {args.config_path}\n")
    f.write("-" * 30 + " Regression & Multiclass Metrics (on all valid samples) " + "-" * 30 + "\n")
    f.write(f"  MAE (Mean Absolute Error): {mae:.4f}\n")
    f.write(f"  Pearson Correlation: {corr:.4f}\n")
    f.write(f"  Multiclass Accuracy (7-class, range {acc7_clip_range}): {acc7:.4f}\n")
    f.write(f"  Multiclass Accuracy (5-class, range {acc5_clip_range}): {acc5:.4f}\n")
    f.write("-" * 30 + " Binary Classification Metrics ('non0' - excluding neutrals - from main head) " + "-" * 30 + "\n")
    f.write(f"  Number of 'non0' samples for binary classification: {num_non0_samples_test}\n")
    f.write(f"  Accuracy (ACC-2, non0): {acc2_non0:.4f}\n")
    f.write(f"  F1-Score (F1, non0, pos_label=1): {f1_non0:.4f}\n")
    f.write("-" * 30 + " Binary Classification Metrics ('has0' - from DEDICATED head) " + "-" * 30 + "\n")
    f.write(f"  Number of 'has0' samples for dedicated binary classification: {num_has0_samples_test_dedicated}\n")
    f.write(f"  Accuracy (ACC-2, has0_dedicated): {acc2_has0_dedicated:.4f}\n")
    f.write(f"  F1-Score (F1, has0_dedicated, pos_label=1): {f1_has0_dedicated:.4f}\n")

print(f"Comprehensive test results saved to {results_output_path}")
print(f"\n--- Summary of Test Metrics (with has0_dedicated head) ---")
print(f"MAE: {mae:.4f}, Corr: {corr:.4f}")
print(f"ACC-2 (non0): {acc2_non0:.4f}, F1 (non0): {f1_non0:.4f} (N={num_non0_samples_test})")
print(f"ACC-2 (has0_dedicated): {acc2_has0_dedicated:.4f}, F1 (has0_dedicated): {f1_has0_dedicated:.4f} (N={num_has0_samples_test_dedicated})")
print(f"ACC-5: {acc5:.4f}, ACC-7: {acc7:.4f}")