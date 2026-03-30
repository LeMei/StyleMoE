import torch
import yaml
import pickle
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import argparse
from tqdm import tqdm

# 确保能从你的项目路径导入模型
from style_moe_multitask_transformer_model import StyleMoEMultiTaskTransformer

# === (1) 关键辅助函数 - 从 test-multitask.py 复制 ===
def valid_seq_mask(x):
    # 检查序列中是否有非零填充
    return (x.abs().sum(dim=-1) > 0).sum(dim=1) >= 1

def clean_inf_nan(tensor):
    # 清理数据中的 inf 和 nan
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    return tensor

def preprocess_test(split_data_dict, p_thresh_non_neutral, n_thresh_non_neutral, p_thresh_has0, text_feature_key='text'):
    # (这个函数与 test-multitask.py 中的完全一致)
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
        # 返回空的，但形状正确的张量
        return torch.empty(0, text.shape[1], text.shape[2]), \
               torch.empty(0, audio.shape[1], audio.shape[2]), \
               torch.empty(0, vision.shape[1], vision.shape[2]), \
               torch.empty(0,1) # 只需要回归标签用于t-SNE

    print(f"Test Split: Total valid samples {len(text_out)}")
    # t-SNE 只需要 text, audio, vision, 和 reg_labels
    return text_out, audio_out, vision_out, reg_labels_out

# === (2) t-SNE 标签分组函数 (来自上次) ===
def create_sentiment_groups(sentiment_scores):
    """
    根据论文 4.4 节中的定义，将连续情感分数转换为离散组。
    (CH-SIMS v2 的阈值)
    0: Strongly Negative (<= -0.5)
    1: Neutral/Subtle (-0.2 到 0.2 之间)
    2: Strongly Positive (>= 0.5)
    3: Other (用于过滤)
    """
    groups = np.full_like(sentiment_scores, 3, dtype=int) # 默认为 "Other"
    groups[sentiment_scores <= -0.5] = 0
    groups[sentiment_scores >= 0.5] = 2
    groups[(sentiment_scores > -0.2) & (sentiment_scores < 0.2)] = 1
    return groups

# === (3) 核心数据生成函数 (已修改) ===
def generate_data(config, test_loader, device, model_path):
    print("正在加载模型...")
    # (修复) 传入完整的 config 字典
    model = StyleMoEMultiTaskTransformer(config)
    
    # --- (修复) 真正加载你训练好的模型权重 ---
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 未找到!")
        print("请确保 config.yaml 中的 output_dir 和 experiment_name 正确，且模型已训练。")
        return
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型权重: {model_path}")
    except RuntimeError as e:
        print(f"加载模型权重时出错: {e}")
        print("请确保你的 .py 模型文件 和 .pth 权重文件结构匹配。")
        return
    
    model.to(device)
    model.eval()

    all_expert_t, all_expert_a, all_expert_v = [], [], []
    all_expert_t2a, all_expert_t2v = [], []
    all_labels_continuous = [] # 收集连续分数

    print("开始遍历测试集收集专家表征...")
    with torch.no_grad():
        # (修复) 修改循环以匹配新的 DataLoader
        for t, a, v, reg_y_b in tqdm(test_loader):
            if not hasattr(t, 'shape') or t.shape[0] == 0: continue
            
            t_d, a_d, v_d, labels_d = \
                [x.to(device) for x in [t, a, v, reg_y_b]]
            
            # --- (4) 调用模型 (使用我们添加的标志) ---
            t_out, a_out, v_out, t2a_out, t2v_out = model(
                t_d, 
                a_d, 
                v_d, 
                return_expert_outputs=True # <--- 关键在这里
            )

            # --- (5) 收集数据 ---
            all_expert_t.append(t_out.cpu())
            all_expert_a.append(a_out.cpu())
            all_expert_v.append(v_out.cpu())
            all_expert_t2a.append(t2a_out.cpu())
            all_expert_t2v.append(t2v_out.cpu())
            all_labels_continuous.append(labels_d.cpu()) # (修复) 收集正确的标签

    print("数据收集完毕。正在合并...")
    # --- (6) 合并并转换为 NumPy (不变) ---
    all_expert_t = torch.cat(all_expert_t, dim=0).numpy()
    all_expert_a = torch.cat(all_expert_a, dim=0).numpy()
    all_expert_v = torch.cat(all_expert_v, dim=0).numpy()
    all_expert_t2a = torch.cat(all_expert_t2a, dim=0).numpy()
    all_expert_t2v = torch.cat(all_expert_t2v, dim=0).numpy()
    all_labels_continuous = torch.cat(all_labels_continuous, dim=0).numpy().flatten() # 确保是一维

    # --- (7) 创建离散的标签组 (不变) ---
    all_labels_grouped = create_sentiment_groups(all_labels_continuous)
    
    # --- (8) 过滤掉不属于任何组的样本 (不变) ---
    mask = (all_labels_grouped != 3)
    print(f"原始样本数: {len(mask)}, 过滤后 (保留 0, 1, 2 组): {mask.sum()}")
    
    # --- (9) 保存到文件 (不变) ---
    output_file = 'tsne_data.npz'
    np.savez_compressed(
        output_file,
        text_expert=all_expert_t[mask],
        audio_expert=all_expert_a[mask],
        video_expert=all_expert_v[mask],
        text_audio_expert=all_expert_t2a[mask],
        text_video_expert=all_expert_t2v[mask],
        labels=all_labels_grouped[mask] 
    )
    print(f"数据已保存到 {output_file}")

# === (5) `main` 函数 - 从 test-multitask.py 借用 ===
if __name__ == "__main__":
    
    # --- (修复) 1. 使用 ArgParse 加载真实 config ---
    parser = argparse.ArgumentParser(description="Export expert representations for t-SNE.")
    parser.add_argument('--config_path', type=str, default='config.yaml', 
                        help='Path to the configuration file (e.g., config.yaml or config_this_run.yaml).')
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"成功加载配置文件: {args.config_path}")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- (修复) 2. 使用 config 加载真实数据 ---
    data_dir = config["paths"]["data_dir"]
    data_file_path = os.path.join(data_dir, f"unaligned.pkl")
    if not os.path.exists(data_file_path):
        data_file_path = os.path.join(data_dir, "unaligned.pkl")
    
    print(f"加载测试数据从: {data_file_path}")
    with open(data_file_path, "rb") as f:
        data_loaded_full = pickle.load(f)

    # (从 test-multitask.py 复制的阈值)
    POSITIVE_THRESHOLD_FOR_BINARY_NON_NEUTRAL = config.get("dataset_specifics", {}).get("positive_threshold_binary_non_neutral", 0.00001) 
    NEGATIVE_THRESHOLD_FOR_BINARY_NON_NEUTRAL = config.get("dataset_specifics", {}).get("negative_threshold_binary_non_neutral", -0.00001)
    POSITIVE_THRESHOLD_FOR_BINARY_HAS0 = config.get("dataset_specifics", {}).get("positive_threshold_binary_has0", 0.0) 
    
    test_text, test_audio, test_vision, test_reg_labels = preprocess_test(
        data_loaded_full["test"], 
        POSITIVE_THRESHOLD_FOR_BINARY_NON_NEUTRAL, 
        NEGATIVE_THRESHOLD_FOR_BINARY_NON_NEUTRAL,
        POSITIVE_THRESHOLD_FOR_BINARY_HAS0,
        text_feature_key=config.get("dataset_specifics",{}).get("text_feature_key_test", 'text')
    )
    
    # (修复) 3. 创建只包含t-SNE所需数据的 DataLoader
    test_loader = DataLoader(
        TensorDataset(test_text, test_audio, test_vision, test_reg_labels),
        batch_size=config["training"].get("batch_size_test", config["training"]["batch_size"])
    )

    # (修复) 4. 找到真实的模型路径
    experiment_name_to_test = config.get("experiment_name", f"{os.path.basename(data_dir)}_multiloss_comprehensive_metrics_has0head")
    model_save_parent_dir = config["paths"]["output_dir"]
    model_save_directory = os.path.join(model_save_parent_dir, experiment_name_to_test)
    model_filename = "best_model_comprehensive_has0head.pth" # 假设使用这个最好的模型
    model_path = os.path.join(model_save_directory, model_filename)

    # --- (6) 运行主函数 ---
    generate_data(config, test_loader, DEVICE, model_path)