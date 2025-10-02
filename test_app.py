# ---------- test_app.py (修正版) ----------
import random
import importlib
import concurrent.futures
import uuid
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


# --- Game configuration ---
DEFAULT_NB_PLAYERS = 6
ROWS, COLS = 20, 30
STAR_COUNT = 50

# Worker pool to safely run user move functions with a timeout
# ThreadPoolExecutorのままで問題ありません
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=DEFAULT_NB_PLAYERS)

# Per-game state (allows many simultaneous viewers/games)
# ★変更点: ゲームごとに星の状態(stars)もサーバーサイドで管理する
# game_id -> {"scores": [...], "nb_players": int, "stars": set((r, c), ...)}
GAMES = {}

# --- Load player move functions (player1.py .. player6.py). Fallback: stay in place ---
move_functions = []
for i in range(1, DEFAULT_NB_PLAYERS + 1):
    module_name = f"player{i}"
    try:
        mod = importlib.import_module(module_name)
        fn = getattr(mod, f"move{i}")
        move_functions.append(fn)
    except Exception:
        # Fallback: keep current position
        def stay(players, stars, idx=i - 1):
            return players[idx]
        move_functions.append(stay)


def generate_initial_positions(nb_players: int):
    """Create random player positions and star tiles."""
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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    """Start a new game instance and return its game_id and initial state."""
    payload = request.get_json(silent=True) or {}
    nb_players = int(payload.get("nbplayers", DEFAULT_NB_PLAYERS))
    nb_players = DEFAULT_NB_PLAYERS

    players, stars = generate_initial_positions(nb_players)
    game_id = uuid.uuid4().hex
    
    # ★変更点: 星の状態をsetとしてサーバーで保持する
    GAMES[game_id] = {
        "scores": [0] * nb_players,
        "nb_players": nb_players,
        "stars": set(tuple(s) for s in stars),
    }
    return jsonify(game_id=game_id, players=players, stars=stars, scores=GAMES[game_id]["scores"])


@app.route("/move", methods=["POST"])
def move():
    """Compute one simulation step with server-side state management."""
    data = request.get_json() or {}
    game_id = data.get("game_id")
    if not game_id or game_id not in GAMES:
        return jsonify(error="invalid game_id"), 400

    players = [tuple(p) for p in data.get("players", [])]
    game_state = GAMES[game_id]
    
    # ★変更点: クライアントから送られた星リストは無視し、サーバーが管理する最新の星の状態を使う
    current_stars_set = game_state["stars"]
    current_stars_list = list(current_stars_set)

    # 1) Ask each move function for a proposed move, with timeout and validation
    proposed = []
    futures = []
    for idx, fn in enumerate(move_functions):
        # ★変更点: AIにはサーバーが管理する最新の星リストを渡す
        future = _executor.submit(fn, players, current_stars_list)
        futures.append(future)

    for idx, future in enumerate(futures):
        try:
            new_coord = future.result(timeout=1.0)
        except Exception:
            new_coord = players[idx]
        
        # Validate: single orthogonal step or stay; inside bounds
        r0, c0 = players[idx]
        r1, c1 = new_coord if isinstance(new_coord, (list, tuple)) and len(new_coord) == 2 else (r0, c0)
        if abs(r1 - r0) + abs(c1 - c0) <= 1 and 0 <= r1 < ROWS and 0 <= c1 < COLS:
            proposed.append((r1, c1))
        else:
            proposed.append((r0, c0))

    # 2) Resolve same-target conflicts randomly
    # (このセクションは変更なし)
    target_to_indices = {}
    for i, tgt in enumerate(proposed):
        target_to_indices.setdefault(tgt, []).append(i)

    winners = set()
    for tgt, idxs in target_to_indices.items():
        if len(idxs) == 1:
            winners.add(idxs[0])
        else:
            winner = random.choice(idxs)
            winners.add(winner)

    # 3) Enforce occupancy rule
    # (このセクションは変更なし)
    will_move = [(i in winners and proposed[i] != players[i]) for i in range(len(proposed))]
    vacating = {players[i]: will_move[i] for i in range(len(players))}

    final_positions = list(players)
    for i in range(len(proposed)):
        if not will_move[i]:
            continue
        target = proposed[i]
        if target in vacating and not vacating[target]:
            continue
        final_positions[i] = target

    # 4) Update score and server-side star state
    scores = game_state["scores"]
    captured_stars = set()

    for idx, pos in enumerate(final_positions):
        # ★変更点: サーバー管理の星セット(current_stars_set)に対してチェック
        if pos in current_stars_set:
            scores[idx] += 1
            captured_stars.add(pos)

    # ★変更点: 取得された星をサーバーの状態から削除
    current_stars_set.difference_update(captured_stars)

    # ★変更点: APIのレスポンスに、更新後の最新の星リストを追加
    return jsonify(coord=final_positions, scores=scores, stars=list(current_stars_set))


if __name__ == "__main__":
    app.run(debug=True)