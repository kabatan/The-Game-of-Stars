import optuna
import subprocess
import json
import os
import multiprocessing as mp

# --- 最適化の設定 ---
PLAYER_ID = 2  # 最適化対象のプレイヤー番号
N_GAMES_PER_TRIAL = 500  # 1回の評価で使用するゲーム数
STUDY_NAME = "p2_aco_optimization" # Optunaの学習履歴の名前
STORAGE_URL = "sqlite:///p2_optimization.db" # 学習履歴を保存するデータベースファイル

# objective関数は前回のものから変更ありません
def objective(trial: optuna.trial.Trial) -> float:
    """Optunaの目的関数。指定されたパラメータで勝率を計算し、返す"""
    
    params = {
        "alpha": trial.suggest_float("alpha", 0.5, 3.0),
        "beta_": trial.suggest_float("beta_", 1.0, 5.0),
        "gamma_": trial.suggest_float("gamma_", 0.5, 3.0),
        "rho": trial.suggest_float("rho", 0.8, 0.99),
        "q": trial.suggest_float("q", 0.5, 2.0),
        "simu_num": trial.suggest_int("simu_num", 5, 20),
        "best_bonus": trial.suggest_float("best_bonus", 1.0, 5.0),
        "win_bonus": trial.suggest_float("win_bonus", 1.0, 5.0),
    }

    print(f"Trial {trial.number}: Testing params: {params}")

    env = os.environ.copy()
    env["PLAYER2_PARAMS"] = json.dumps(params)
    
    command = [
        "python", 
        "game_statistics.py",
        "--num-games", str(N_GAMES_PER_TRIAL),
        "--json-output"
    ]
    
    try:
        result = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        stats = json.loads(result.stdout)
        win_rate = stats["win_rates"][f"player_{PLAYER_ID}"]
        print(f"Trial {trial.number}: Player {PLAYER_ID} Win Rate = {win_rate:.2f}%")
        
        return win_rate

    except subprocess.CalledProcessError as e:
        print(f"Error in trial {trial.number}:")
        print(e.stderr)
        return 0.0
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Failed to parse output in trial {trial.number}: {e}")
        return 0.0

def main():
    n_jobs = max(1, mp.cpu_count() // 4)
    # ★変更点: 事前に試行回数を決めないため、メッセージを変更
    print(f"Starting continuous optimization. Press Ctrl+C to stop.")
    print(f"Optuna will run {n_jobs} trials in parallel.")
    print(f"Each trial will simulate {N_GAMES_PER_TRIAL} games.")
    print(f"Study name: '{STUDY_NAME}', Storage: '{STORAGE_URL}'")

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        load_if_exists=True,
        direction="maximize"
    )
    
    try:
        # ★変更点: n_trials=None に設定し、無限に学習を続ける
        study.optimize(objective, n_trials=None, n_jobs=n_jobs)
    except KeyboardInterrupt:
        # ★変更点: Ctrl+C が押されたら、ここが実行される
        print("\nOptimization stopped by user.")
    
    print("\n--- Current Best Results ---")
    # 停止した時点での最良の結果を表示
    # study.best_trial はDBから読み込むため、最新の情報が反映される
    # トライアルが一度も成功していない場合はエラーになるためチェック
    if study.best_trial:
        print(f"Best trial number: {study.best_trial.number}")
        print(f"Best win rate: {study.best_value:.2f}%")
        print("Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
    else:
        print("No completed trials found.")

if __name__ == "__main__":
    main()