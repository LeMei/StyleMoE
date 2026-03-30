import torch
import torch.nn.functional as F
import yaml
import pickle
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns # For better plotting aesthetics
import argparse

# --- Import necessary components from your model and test scripts ---
# Make sure these files are in the same directory or accessible via PYTHONPATH
from style_moe_multitask_transformer_model import StyleMoEMultiTaskTransformer
# Copy necessary preprocessing functions from test-multitask.py
def valid_seq_mask(x):
    # Ensure tensor has at least 2 dimensions for sum
    if x.dim() < 2:
        return torch.tensor(False) # Or handle based on expected input
    # Check if sum dim exists (-1 usually works for both [B, D] and [B, S, D])
    if x.shape[-1] == 0: return torch.zeros(x.shape[0], dtype=torch.bool, device=x.device) # Handle empty feature dim
    dim_to_sum = -1 if x.dim() >= 2 else 0 # Adjust dim if needed

    # Sum over the feature dimension first
    sum_over_features = x.abs().sum(dim=dim_to_sum)
    # If original was 3D [B, S, D], sum_over_features is [B, S] -> sum over S
    # If original was 2D [B, D], sum_over_features is [B] -> check if > 0
    if sum_over_features.dim() > 1: # Assumes [B, S]
         sum_over_seq = sum_over_features.sum(dim=1)
    else: # Assumes [B]
         sum_over_seq = sum_over_features

    # Final check: At least one element in the sequence (if applicable) or the vector itself must be non-zero
    # And have at least one valid timestep if sequence data
    min_valid_steps = 1
    has_min_steps = True
    if x.dim() > 2 and x.shape[1] > 0: # Check sequence length if 3D tensor
        has_min_steps = (x.abs().sum(dim=-1) > 0).sum(dim=1) >= min_valid_steps


    return (sum_over_seq > 0) & has_min_steps


def clean_inf_nan(tensor):
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    return tensor

def preprocess_test(split_data_dict, config, p_thresh_non_neutral, n_thresh_non_neutral, p_thresh_has0, text_feature_key='text'):
    # This function is assumed to be identical to the one in test-multitask.py
    # ... (Copy the full function implementation here) ...
    # Ensure it returns: text_out, audio_out, vision_out, reg_labels_out,
    #                   binary_cls_labels_non0_out, non_neutral_mask_out, binary_cls_labels_has0_out
    # (Copied from previous context for completeness)
    text_data = split_data_dict.get(text_feature_key, split_data_dict.get('text', split_data_dict.get('words', split_data_dict.get('text_features'))))
    audio_data = split_data_dict.get('audio', split_data_dict.get('audio_features'))
    vision_data = split_data_dict.get('vision', split_data_dict.get('vision_features'))
    reg_labels_data = split_data_dict.get('regression_labels', split_data_dict.get('labels'))

    # Check if data is already tensor (might happen if loaded differently)
    text = text_data if isinstance(text_data, torch.Tensor) else torch.tensor(text_data, dtype=torch.float32)
    audio = audio_data if isinstance(audio_data, torch.Tensor) else torch.tensor(audio_data, dtype=torch.float32)
    vision = vision_data if isinstance(vision_data, torch.Tensor) else torch.tensor(vision_data, dtype=torch.float32)
    original_reg_labels = reg_labels_data if isinstance(reg_labels_data, torch.Tensor) else torch.tensor(reg_labels_data, dtype=torch.float32)

    text = clean_inf_nan(text)
    audio = clean_inf_nan(audio)
    vision = clean_inf_nan(vision)

    if original_reg_labels.ndim == 3:
        original_reg_labels = original_reg_labels.squeeze(-1)
    original_reg_labels = original_reg_labels.view(-1, 1)

    # Make sure valid_seq_mask works correctly even for single samples during preprocessing
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
        # Ensure correct dimensionality based on actual data structure (e.g., [B, S, D] vs [B, D])
        seq_dim_t = text.shape[1] if text.dim() > 2 else 0
        seq_dim_a = audio.shape[1] if audio.dim() > 2 else 0
        seq_dim_v = vision.shape[1] if vision.dim() > 2 else 0

        text_empty = torch.empty((0, seq_dim_t, text_shape_orig[-1]) if seq_dim_t else (0, text_shape_orig[-1]))
        audio_empty = torch.empty((0, seq_dim_a, audio_shape_orig[-1]) if seq_dim_a else (0, audio_shape_orig[-1]))
        vision_empty = torch.empty((0, seq_dim_v, vision_shape_orig[-1]) if seq_dim_v else (0, vision_shape_orig[-1]))

        return text_empty, audio_empty, vision_empty, \
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
# --- End of copied functions ---

# --- Main Script Logic ---
def analyze_weights(config_path):
    # 1. Load Config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"成功加载配置文件: {config_path}")

    # Derive paths from config
    experiment_name_to_test = config.get("experiment_name", "ch-sims-v2_multiloss_comprehensive_metrics_has0head") # Use default if missing
    model_save_parent_dir = config["paths"]["output_dir"]
    model_save_directory = os.path.join(model_save_parent_dir, experiment_name_to_test)
    model_filename = config.get("testing", {}).get("model_filename", "best_model_comprehensive_has0head.pth")
    model_path = os.path.join(model_save_directory, model_filename)
    data_dir = config["paths"]["data_dir"]
    dataset_name = os.path.basename(data_dir)

    # Check if necessary files exist
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 未找到!")
        exit(1)

    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Load and Preprocess Data
    data_file_path = os.path.join(data_dir, f"unaligned.pkl")
    # Add fallback logic as in test-multitask.py
    if not os.path.exists(data_file_path):
        data_file_path_alt = os.path.join(data_dir, "unaligned.pkl")
        if os.path.exists(data_file_path_alt):
            data_file_path = data_file_path_alt
        else:
            print(f"错误: 测试数据文件 {data_file_path} 或备选路径均未找到!")
            exit(1)

    print(f"Loading test data from: {data_file_path}")
    with open(data_file_path, "rb") as f:
        data_loaded_full = pickle.load(f)

    if 'test' not in data_loaded_full:
        print(f"错误: 加载的数据中未找到 'test' split。")
        exit(1)

    # Use thresholds defined in config (same as testing script)
    p_thresh_non0 = config.get("dataset_specifics", {}).get("positive_threshold_binary_non_neutral", 0.00001)
    n_thresh_non0 = config.get("dataset_specifics", {}).get("negative_threshold_binary_non_neutral", -0.00001)
    p_thresh_has0 = config.get("dataset_specifics", {}).get("positive_threshold_binary_has0", 0.0)

    test_text, test_audio, test_vision, test_reg_labels, _, _, _ = preprocess_test(
        data_loaded_full["test"], config, p_thresh_non0, n_thresh_non0, p_thresh_has0,
        text_feature_key=config.get("dataset_specifics",{}).get("text_feature_key_test", 'text')
    )

    if test_text.numel() == 0:
        print("错误: 测试集预处理后没有有效样本。")
        exit()

    # Use a larger batch size for inference if possible, defined in config or default
    test_batch_size = config["training"].get("batch_size_test", config["training"]["batch_size"])
    test_loader = DataLoader(
        TensorDataset(test_text, test_audio, test_vision, test_reg_labels),
        batch_size=test_batch_size
    )
    print(f"Test loader created with batch size: {test_batch_size}")

    # 4. Load Model
    model = StyleMoEMultiTaskTransformer(config).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights from {model_path}")
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        exit(1)
    model.eval() # Set to evaluation mode

    # 5. Inference and Weight Extraction
    all_router_weights = []
    all_reg_labels = []

    print("Starting inference to extract router weights...")
    with torch.no_grad():
        for i, (t, a, v, reg_y_b) in enumerate(test_loader):
            if not hasattr(t, 'shape') or t.shape[0] == 0: continue

            t_d, a_d, v_d, reg_y_d = [x.to(device) for x in [t, a, v, reg_y_b]]

            # --- Replicate the relevant parts of the forward pass ---
            # Get unimodal features
            t_feat = model.text_encoder(t_d)
            a_feat = model.audio_encoder(a_d)
            v_feat = model.vision_encoder(v_d)

            # Get cross-modal features (use original sequences before dropout for consistency if needed)
            # Assuming a_d and v_d here are the sequences passed to the loader
            t2a = model.cross_text_audio(t_feat.unsqueeze(1), a_d)
            t2v = model.cross_text_vision(t_feat.unsqueeze(1), v_d)

            # Construct router input
            router_input = torch.cat([t_feat, a_feat, v_feat, t2a, t2v], dim=-1)

            # Get router weights
            weights = F.softmax(model.router(router_input), dim=-1)
            # --- End replication ---

            all_router_weights.append(weights.cpu().numpy())
            all_reg_labels.append(reg_y_d.cpu().numpy())
            
            if (i + 1) % 50 == 0: # Print progress
                 print(f"Processed batch {i+1}/{len(test_loader)}")


    print("Inference complete.")
    # Concatenate results from all batches
    all_router_weights_np = np.concatenate(all_router_weights, axis=0)
    all_reg_labels_np = np.concatenate(all_reg_labels, axis=0).flatten() # Ensure labels are 1D

    print(f"Total weights shape: {all_router_weights_np.shape}") # Should be [num_samples, 5]
    print(f"Total labels shape: {all_reg_labels_np.shape}")   # Should be [num_samples]

    # 6. Group Samples by Sentiment
    # Define thresholds exactly as in the target figure's description
    strongly_pos_mask = all_reg_labels_np >= 0.5
    strongly_neg_mask = all_reg_labels_np <= -0.5
    neutral_subtle_mask = (all_reg_labels_np > -0.2) & (all_reg_labels_np < 0.2)

    # 7. Calculate Average Weights per Group
    expert_names = ["Text", "Audio", "Video", "Text-Audio", "Text-Video"]
    avg_weights = {}

    weights_strong_pos = all_router_weights_np[strongly_pos_mask]
    weights_strong_neg = all_router_weights_np[strongly_neg_mask]
    weights_neutral = all_router_weights_np[neutral_subtle_mask]

    print(f"Num Strongly Positive: {weights_strong_pos.shape[0]}")
    print(f"Num Strongly Negative: {weights_strong_neg.shape[0]}")
    print(f"Num Neutral/Subtle: {weights_neutral.shape[0]}")

    if weights_strong_pos.shape[0] > 0:
        avg_weights["Strongly Positive"] = np.mean(weights_strong_pos, axis=0)
    else: avg_weights["Strongly Positive"] = np.zeros(len(expert_names))

    if weights_strong_neg.shape[0] > 0:
        avg_weights["Strongly Negative"] = np.mean(weights_strong_neg, axis=0)
    else: avg_weights["Strongly Negative"] = np.zeros(len(expert_names))

    if weights_neutral.shape[0] > 0:
        avg_weights["Neutral/Subtle"] = np.mean(weights_neutral, axis=0)
    else: avg_weights["Neutral/Subtle"] = np.zeros(len(expert_names))


    print("\n--- Average Router Weights per Group ---")
    print(f"Groups: {list(avg_weights.keys())}")
    print(f"Experts: {expert_names}")
    for group, weights in avg_weights.items():
        print(f"{group}: {[f'{w:.3f}' for w in weights]}")

    # 8. Visualize
    sns.set_theme(style="whitegrid")
    n_groups = len(avg_weights)
    n_experts = len(expert_names)
    index = np.arange(n_experts)
    bar_width = 0.25 # Adjust as needed

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = sns.color_palette("viridis", n_groups) # Or choose another palette

    for i, (group, weights) in enumerate(avg_weights.items()):
        ax.bar(index + i * bar_width, weights, bar_width, label=group, color=colors[i])

    ax.set_xlabel('Expert Network', fontweight='bold')
    ax.set_ylabel('Average Weight', fontweight='bold')
    ax.set_title(f'Average Router Weight Distribution by Sentiment Group ({dataset_name.upper()})', fontweight='bold', fontsize=14)
    ax.set_xticks(index + bar_width * (n_groups - 1) / 2)
    ax.set_xticklabels(expert_names)
    ax.legend(title="Sentiment Group")
    ax.set_ylim(0, max(np.max(list(avg_weights.values())) * 1.1, 0.1)) # Adjust y-limit
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save the figure in the model's output directory
    plot_filename = "router_weight_distribution_analysis.png"
    plot_save_path = os.path.join(model_save_directory, plot_filename)
    plt.savefig(plot_save_path, dpi=300)
    print(f"\nRouter weight distribution plot saved to: {plot_save_path}")
    # plt.show() # Uncomment to display plot interactively


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze router weights for a trained StyleMoE model.")
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the specific run configuration file (e.g., outputs/.../config_this_run.yaml).')
    args = parser.parse_args()
    analyze_weights(args.config_path)