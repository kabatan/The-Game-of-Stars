import importlib
import concurrent.futures
import numpy as np
import time
from collections import defaultdict
from typing import List, Dict, Any
import multiprocessing as mp
import sys
import argparse
import json

# C++で実装した高速シミュレータをインポート
import game_simulator_cpp

# --- Game configuration (C++側と共通) ---
DEFAULT_NB_PLAYERS = 6

def _load_move_functions():
    """プレイヤーのmove関数を読み込む"""
    move_functions = []
    for i in range(1, DEFAULT_NB_PLAYERS + 1):
        module_name = f"player{i}"
        try:
            if module_name in sys.modules:
                mod = importlib.reload(sys.modules[module_name])
            else:
                mod = importlib.import_module(module_name)
            fn = getattr(mod, f"move{i}")
            move_functions.append(fn)
        except Exception:
            # フォールバック：その場に留まる関数
            def stay(players, stars, idx=i-1):
                return players[idx]
            move_functions.append(stay)
    return move_functions

def simulate_single_game(game_id: int) -> Dict[str, Any]:
    """
    C++シミュレータを呼び出すためのラッパー関数。
    各プロセスでプレイヤーモジュールを読み込み、C++に渡す。
    """
    # ワーカプロセスごとにmove関数をロードする
    move_functions = _load_move_functions()
    
    # C++関数を呼び出し、結果をPython辞書として受け取る
    result = game_simulator_cpp.simulate_single_game_cpp(move_functions)
    result['game_id'] = game_id # game_idを結果に追加
    return result

# --- run_parallel_simulations, analyze_results, main関数は以前のものから変更なし ---

def run_parallel_simulations(num_games: int, num_processes: int = None) -> List[Dict[str, Any]]:
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 16)
    
    is_json_output = "--json-output" in sys.argv
    if not is_json_output:
        print(f"Running {num_games} C++ simulations with {num_processes} processes...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # 呼び出す関数を、C++ラッパーであるsimulate_single_gameに変更
        futures = [executor.submit(simulate_single_game, i) for i in range(num_games)]
        results = []
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"A simulation task failed: {e}", file=sys.stderr)

            completed += 1
            if not is_json_output and completed % max(1, num_games // 10) == 0:
                print(f"Progress: {completed}/{num_games} ({completed/num_games*100:.1f}%)", file=sys.stderr)
    
    return results

def analyze_results(results: List[Dict[str, Any]], json_output: bool = False) -> Dict[str, Any]:
    if not results:
        print("No results to analyze.", file=sys.stderr)
        return {}
    
    wins_by_player = defaultdict(int)
    ties = 0
    
    for result in results:
        if result.get('is_tie', False):
            ties += 1
        elif result.get('winner', -1) != -1:
            wins_by_player[result['winner']] += 1

    output_data = {"num_games": len(results), "win_rates": {}, "ties": {}}
    for player in range(DEFAULT_NB_PLAYERS):
        wins = wins_by_player[player]
        win_rate = wins / len(results) * 100 if results else 0
        output_data["win_rates"][f"player_{player + 1}"] = win_rate
    
    output_data["ties"]["count"] = ties
    output_data["ties"]["rate"] = ties / len(results) * 100 if results else 0
    
    if json_output:
        print(json.dumps(output_data))
    else:
        print(f"\n=== Analysis of {len(results)} games ===")
        print("\n--- Win Rate Statistics ---")
        for player in range(DEFAULT_NB_PLAYERS):
            print(f"Player {player + 1}: {wins_by_player[player]:4d} wins ({output_data['win_rates'][f'player_{player + 1}']:5.1f}%)")
        print(f"Ties:         {ties:4d} games ({output_data['ties']['rate']:5.1f}%)")

    return output_data

def main():
    parser = argparse.ArgumentParser(description="Game Statistics Analyzer (C++ Backend)")
    parser.add_argument("-n", "--num-games", type=int, default=1000, help="Number of games to simulate")
    parser.add_argument("-p", "--num-processes", type=int, default=None, help=f"Number of processes to use (default: auto, max {mp.cpu_count()})")
    parser.add_argument("--json-output", action="store_true", help="Output analysis results in JSON format to stdout")
    args = parser.parse_args()

    if not args.json_output:
        print("Game Statistics Analyzer (C++ Backend)")
        print("=" * 50)

    start_time = time.time()
    
    try:
        results = run_parallel_simulations(args.num_games, args.num_processes)
        end_time = time.time()
        duration = end_time - start_time
        
        if not args.json_output:
            games_per_second = args.num_games / duration if duration > 0 else 0
            print(f"\nSimulation completed in {duration:.2f} seconds")
            print(f"Rate: {games_per_second:.1f} games/second")
            
        analyze_results(results, json_output=args.json_output)
    
    except Exception as e:
        print(f"Error during simulation: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()