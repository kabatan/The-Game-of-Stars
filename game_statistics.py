import random
import importlib
import concurrent.futures
import numpy as np
import time
from collections import defaultdict
from typing import List, Tuple, Dict, Any
import multiprocessing as mp
import sys
import argparse
import json

# --- Game configuration ---
DEFAULT_NB_PLAYERS = 6
ROWS, COLS = 20, 30
STAR_COUNT = 50
MAX_TURNS = 200  # ゲーム終了条件

class GameSimulator:
    def __init__(self):
        self.move_functions = self._load_move_functions()
    
    def _load_move_functions(self):
        """プレイヤーのmove関数を読み込み (reload each time to reset globals)"""
        move_functions = []
        for i in range(1, DEFAULT_NB_PLAYERS + 1):
            module_name = f"player{i}"
            try:
                if module_name in sys.modules:
                    # force reload if already imported
                    mod = importlib.reload(sys.modules[module_name])
                else:
                    mod = importlib.import_module(module_name)
                fn = getattr(mod, f"move{i}")
                move_functions.append(fn)
            except Exception:
                def stay(players, stars, idx=i-1):
                    return players[idx]
                move_functions.append(stay)
        return move_functions
    
    def generate_initial_positions(self, nb_players: int):
        """初期位置とスター配置を生成"""
        players = []
        while len(players) < nb_players:
            r, c = random.randint(0, ROWS - 1), random.randint(0, COLS - 1)
            if all(abs(r - pr) + abs(c - pc) >= 5 for pr, pc in players):
                players.append((r, c))

        stars = []
        while len(stars) < STAR_COUNT:
            r, c = random.randint(0, ROWS - 1), random.randint(0, COLS - 1)
            if (r, c) not in stars and all(abs(r - pr) + abs(c - pc) > 4 for pr, pc in players):
                stars.append((r, c))
        
        return players, stars
    
    def simulate_turn(self, players: List[Tuple[int, int]], stars: List[Tuple[int, int]], 
                      scores: List[int]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[int]]:
        """1ターンをシミュレート"""
        players_before_turn = list(players)
        
        proposed = []
        for idx, fn in enumerate(self.move_functions):
            try:
                new_coord = fn(players, stars)
            except Exception:
                new_coord = players[idx]
            
            # 移動の妥当性チェック
            r0, c0 = players[idx]
            r1, c1 = new_coord if isinstance(new_coord, (list, tuple)) and len(new_coord) == 2 else (r0, c0)
            if abs(r1 - r0) + abs(c1 - c0) <= 1 and 0 <= r1 < ROWS and 0 <= c1 < COLS:
                proposed.append((r1, c1))
            else:
                proposed.append((r0, c0))
        
        old_positions = players[:]
        
        # 同じターゲットへの衝突解決
        target_to_indices = defaultdict(list)
        for i, tgt in enumerate(proposed):
            target_to_indices[tgt].append(i)
        
        winners = set()
        for tgt, idxs in target_to_indices.items():
            if len(idxs) == 1:
                winners.add(idxs[0])
            else:
                winners.add(random.choice(idxs))
        
        # 占有ルールの適用
        will_move = [(i in winners) and (proposed[i] != old_positions[i]) for i in range(len(proposed))]
        vacating = {old_positions[i]: will_move[i] for i in range(len(old_positions))}
        
        final_positions = list(old_positions)
        for i in range(len(proposed)):
            if not will_move[i]:
                continue
            target = proposed[i]
            if target in vacating and not vacating[target]:
                continue
            final_positions[i] = target
        
        # スコア更新とスター削除
        new_scores = scores[:]
        star_set = set(stars)
        collected_stars = set()
        
        for idx, pos in enumerate(final_positions):
            if pos in star_set:
                new_scores[idx] += 1
                collected_stars.add(pos)
        
        new_stars = [star for star in stars if star not in collected_stars]

        # --- デバッグ用出力（標準エラー出力へ） ---
        player2_before = players_before_turn[1]
        player2_after = final_positions[1]
        score2_before = scores[1]
        score2_after = new_scores[1]

        if player2_before != player2_after:
            print(f"[DEBUG] Player 2 moved: {player2_before} -> {player2_after}", file=sys.stderr)
        if score2_before != score2_after:
            print(f"[DEBUG] Player 2 scored! Score: {score2_before} -> {score2_after}", file=sys.stderr)
        # --- デバッグここまで ---
        
        return final_positions, new_stars, new_scores
    
    def simulate_game(self, game_id: int = 0) -> Dict[str, Any]:
        """1ゲーム全体をシミュレート"""
        self.move_functions = self._load_move_functions()

        players, stars = self.generate_initial_positions(DEFAULT_NB_PLAYERS)
        scores = [0] * DEFAULT_NB_PLAYERS
        turn = 0
        
        while turn < MAX_TURNS and len(stars) > 0:
            players, stars, scores = self.simulate_turn(players, stars, scores)
            turn += 1
        
        max_score = max(scores) if scores else 0
        winners = [i for i, score in enumerate(scores) if score == max_score]
        
        return {
            'game_id': game_id,
            'scores': scores,
            'turns': turn,
            'winner': winners[0] if len(winners) == 1 else -1,
            'is_tie': len(winners) > 1,
            'max_score': max_score,
            'total_stars_collected': sum(scores),
            'remaining_stars': len(stars)
        }

def run_parallel_simulations(num_games: int, num_processes: int = None) -> List[Dict[str, Any]]:
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 16)
    
    # JSON出力モードでない場合のみ、進捗状況を表示
    is_json_output = "--json-output" in sys.argv
    if not is_json_output:
        print(f"Running {num_games} simulations with {num_processes} processes...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
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

def simulate_single_game(game_id: int) -> Dict[str, Any]:
    simulator = GameSimulator()
    return simulator.simulate_game(game_id)

def analyze_results(results: List[Dict[str, Any]], json_output: bool = False) -> Dict[str, Any]:
    if not results:
        print("No results to analyze.", file=sys.stderr)
        return {}
    
    wins_by_player = defaultdict(int)
    total_scores_by_player = defaultdict(list)
    ties = 0
    
    for result in results:
        if result['is_tie']:
            ties += 1
        else:
            wins_by_player[result['winner']] += 1
        
        for i, score in enumerate(result['scores']):
            total_scores_by_player[i].append(score)

    # --- JSON出力用のデータを作成 ---
    output_data = {
        "num_games": len(results),
        "win_rates": {},
        "avg_scores": {},
        "ties": {
            "count": ties,
            "rate": ties / len(results) * 100 if results else 0
        }
    }
    for player in range(DEFAULT_NB_PLAYERS):
        wins = wins_by_player[player]
        win_rate = wins / len(results) * 100 if results else 0
        scores = total_scores_by_player[player]
        avg_score = np.mean(scores) if scores else 0
        
        output_data["win_rates"][f"player_{player + 1}"] = win_rate
        output_data["avg_scores"][f"player_{player + 1}"] = avg_score
    
    if json_output:
        # JSONフラグがTrueなら、結果をJSON形式で標準出力に出力
        print(json.dumps(output_data))
    else:
        # 従来通りのコンソール表示
        print(f"\n=== Analysis of {len(results)} games ===")
        print("\n--- Win Rate Statistics ---")
        for player in range(DEFAULT_NB_PLAYERS):
            print(f"Player {player + 1}: {wins_by_player[player]:4d} wins ({output_data['win_rates'][f'player_{player + 1}']:5.1f}%)")
        print(f"Ties:         {ties:4d} games ({output_data['ties']['rate']:5.1f}%)")
        # ... (他の統計情報の表示もここに追加可能) ...

    return output_data

def main():
    # --- コマンドライン引数のパーサーを設定 ---
    parser = argparse.ArgumentParser(description="Game Statistics Analyzer")
    parser.add_argument(
        "-n", "--num-games", type=int, default=1000,
        help="Number of games to simulate"
    )
    parser.add_argument(
        "-p", "--num-processes", type=int, default=None,
        help=f"Number of processes to use (default: auto, max {mp.cpu_count()})"
    )
    parser.add_argument(
        "--json-output", action="store_true",
        help="Output analysis results in JSON format to stdout"
    )
    args = parser.parse_args()

    if not args.json_output:
        print("Game Statistics Analyzer")
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
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.", file=sys.stderr)
    except Exception as e:
        print(f"Error during simulation: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()