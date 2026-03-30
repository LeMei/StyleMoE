import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time

def plot_tsne_combined_high_res():
    input_file = 'tsne_data.npz'
    print(f"正在加载数据: {input_file}...")
    try:
        data = np.load(input_file)
    except FileNotFoundError:
        print(f"错误: 未找到 {input_file}。请先运行 generate_tsne_data.py。")
        return

    expert_keys = [
        'text_expert', 'audio_expert', 'video_expert', 
        'text_audio_expert', 'text_video_expert'
    ]
    # 图例名称
    titles = [
        "Text Expert", "Audio Expert", "Video Expert", 
        "Text-Audio Expert", "Text-Video Expert"
    ]
    
    all_data_list = []
    all_labels_list = []
    
    for i, key in enumerate(expert_keys):
        expert_data = data[key]
        num_samples = expert_data.shape[0]
        
        all_data_list.append(expert_data)
        expert_labels = np.full(num_samples, i)
        all_labels_list.append(expert_labels)
    
    combined_data = np.concatenate(all_data_list, axis=0)
    combined_labels = np.concatenate(all_labels_list, axis=0)
    
    print(f"总样本数 (5 * N_samples): {combined_data.shape[0]}")
    
    print("正在运行 t-SNE... (这可能需要几分钟)")
    tsne = TSNE(n_components=2, 
                perplexity=50, 
                n_iter=1000, 
                random_state=42, 
                n_jobs=-1) 
                
    start_time = time.time()
    tsne_results = tsne.fit_transform(combined_data)
    print(f"t-SNE 耗时: {time.time() - start_time:.2f} 秒")

    # --- 绘图设置 ---

    # 1. 字体设置 (Times New Roman)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['xtick.labelsize'] = 12 # 刻度数字大小
    plt.rcParams['ytick.labelsize'] = 12

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 10)) 

    # --- 2. 颜色设置 (保持不变) ---
    custom_colors = [
        '#9ECBE3',  # 浅蓝
        '#C56C78',  # 砖红
        '#EAD787',  # 鹅黄
        '#D0B0D1',  # 浅紫
        '#90CD96'   # 柔和绿
    ]
    
    for i, (title, color) in enumerate(zip(titles, custom_colors)):
        mask = (combined_labels == i)
        
        ax.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                   c=[color], 
                   label=title, 
                   s=20,           
                   alpha=0.8,      
                   edgecolors='none') 
    
    # --- 3. 坐标轴设置 (修改点) ---
    # 只置空 Label，保留刻度数字
    ax.set_xlabel("") 
    ax.set_ylabel("")
    # ax.set_xticks([]) # 这行删掉，保留X轴数字
    # ax.set_yticks([]) # 这行删掉，保留Y轴数字

    # --- 4. 边框设置 (全包边框) ---
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)

    # --- 5. 图例设置 (修改点：位置改为左下角) ---
    ax.legend(markerscale=3, loc='lower left', frameon=True, edgecolor='black')    

    output_image = "tsne_experts_styled_v2.png"
    plt.savefig(output_image, bbox_inches='tight', dpi=300) 
    print(f"\n图像已保存到 {output_image}")
    plt.show()

if __name__ == "__main__":
    plot_tsne_combined_high_res()