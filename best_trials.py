import optuna

# 最適化で使ったDBファイルとStudy名を指定
STORAGE_URL = "sqlite:///p2_optimization.db"
STUDY_NAME = "p2_aco_optimization" # optimizer.pyと同じ名前にする

# 上位何件を候補としてリストアップするか
N_TOP_TRIALS = 10

# 学習履歴を読み込む
study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)

# 評価値（勝率）が高かった順にトライアルを取得
best_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
best_trials.sort(key=lambda t: t.value, reverse=True)

print(f"--- Top {N_TOP_TRIALS} Best Trials ---")

for i, trial in enumerate(best_trials[:N_TOP_TRIALS]):
    print(f"\n--- Rank {i+1} ---")
    print(f"Trial Number: {trial.number}")
    print(f"Win Rate (in optimization): {trial.value:.2f}%")
    print("Parameters:")
    # パラメータをC++に書き込みやすい形式で出力
    for key, value in trial.params.items():
        if isinstance(value, int):
            print(f"    {key} = {value};")
        else:
            print(f"    {key} = {value}f;") # float型はfを付けるとC++で分かりやすい