import optuna

# --- 请根据您的设置修改下面三行 ---
STUDY_NAME = "sims-v2_has0head_v1"
# 请将这里的路径替换为您 .db 文件的真实绝对路径
DB_STORAGE_PATH = "sqlite:///D:/Study/MSC/thesis/model_code/model/outputs/ch-sims-v2/optuna_study_sims-v2_has0head_v2/sims-v2_has0head_v1.db" 
TRIAL_NUMBER = 93
# -----------------------------------

try:
    print(f"正在加载 study '{STUDY_NAME}' 从 '{DB_STORAGE_PATH}'...")
    study = optuna.load_study(study_name=STUDY_NAME, storage=DB_STORAGE_PATH)
    
    # Optuna 的 trial 列表是从0开始索引的，所以第93轮是 study.trials[93]
    if len(study.trials) > TRIAL_NUMBER:
        trial = study.trials[TRIAL_NUMBER]
        print("\n" + "="*40)
        print(f"--- Trial {TRIAL_NUMBER} 的真实参数 ---")
        print(trial.params)
        print("="*40)
    else:
        print(f"错误：Study 中没有第 {TRIAL_NUMBER} 轮 trial。请检查 TRIAL_NUMBER 是否正确。")

except Exception as e:
    print(f"加载 study 时出错: {e}")
    print("请确认 DB_STORAGE_PATH 是否正确，并且文件存在。")