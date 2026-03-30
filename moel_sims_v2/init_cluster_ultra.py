import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
data_dir = config["paths"]["data_dir"]

with open(os.path.join(data_dir, "unaligned.pkl"), "rb") as f:
    data = pickle.load(f)

text = data['train']['text'].mean(axis=1)      # [N, 768]
audio = data['train']['audio'].mean(axis=1)    # [N, 25]
vision = data['train']['vision'].mean(axis=1)  # [N, 177]

# 1. 过滤掉包含NaN/Inf的样本
def is_good(x):
    return np.all(np.isfinite(x)) and not np.allclose(x, 0)

mask = np.array([is_good(t) and is_good(a) and is_good(v) for t, a, v in zip(text, audio, vision)])
print("有效样本数: ", np.sum(mask), "/", len(mask))
text, audio, vision = text[mask], audio[mask], vision[mask]

# 2. PCA
text_feat = PCA(n_components=128, random_state=42).fit_transform(text)
audio_feat = audio
vision_feat = vision

features = np.concatenate([text_feat, audio_feat, vision_feat], axis=1)  # [N, 192]

# 3. 补零到256维
if features.shape[1] < 256:
    tmp = np.zeros((features.shape[0], 256), dtype=np.float32)
    tmp[:, :features.shape[1]] = features
    features = tmp

# 4. 检查极值
print("features max/min:", np.max(features), np.min(features))
print("features contains NaN:", np.isnan(features).any())
print("features contains Inf:", np.isinf(features).any())

# 5. 修复极端值
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

# 6. KMeans
num_experts = 5
kmeans = KMeans(n_clusters=num_experts, random_state=42, n_init=10)
kmeans.fit(features)
centers = kmeans.cluster_centers_   # shape: (5, 256)
np.save('centers_256dim_sims-v2.npy', centers)
print(f"[INFO] 已保存聚类中心: {centers.shape} 到 centers_256dim_sims-v2.npy")

