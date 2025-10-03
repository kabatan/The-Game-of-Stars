# ---------- test_app.py (改定後) ----------
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
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=DEFAULT_NB_PLAYERS)

# Per-game state (allows many simultaneous viewers/games)
# game_id -> {"scores": [...], "nb_players": int, "stars": set((r, c), ...)}
GAMES = {}

# --- Load player move functions and modules ---
# ★変更点: move_functions と一緒に、モジュール自体も保持する
move_functions = []
player_modules = []  # 各player*.pyモジュールを格納するリスト
for i in range(1, DEFAULT_NB_PLAYERS + 1):
    module_name = f"player{i}"
    try:
        mod = importlib.import_module(module_name)
        fn = getattr(mod, f"move{i}")
        move_functions.append(fn)
        player_modules.append(mod)  # モジュールをリストに追加
    except Exception:
        # Fallback: keep current position
        def stay(players, stars, idx=i - 1):
            return players[idx]
        move_functions.append(stay)
        player_modules.append(None) # モジュールがない場合はNoneを追加


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
    nb_players = DEFAULT_NB_PLAYERS # Fixed to default for now

    players, stars = generate_initial_positions(nb_players)
    game_id = uuid.uuid4().hex
    
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
    
    current_stars_set = game_state["stars"]
    current_stars_list = list(current_stars_set)

    # 1) Ask each move function for a proposed move, with timeout and validation
    proposed = []
    futures = []
    for idx, fn in enumerate(move_functions):
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
        if pos in current_stars_set:
            scores[idx] += 1
            captured_stars.add(pos)
    
    current_stars_set.difference_update(captured_stars)

    # 5) ★新規追加: 各ボットから思考中のルート情報を安全に取得する
    all_routes = []
    for i, mod in enumerate(player_modules):
        route_coords = []  # デフォルトは空のルート
        
        # モジュールが存在し、必要な属性を持っているか安全にチェック
        if mod:
            try:
                # 1. bot_state を取得。なければ AttributeError
                bot_state = getattr(mod, 'bot_state')
                
                # 2. 必要なリストを取得。なければ KeyError
                star_list = bot_state['star_list']
                all_player_routes = bot_state['global_best_route']
                
                # 3. 対象プレイヤーのルート(インデックスのリスト)を取得
                #    インデックスが範囲外なら IndexError, 型が違えば TypeError
                route_indices = all_player_routes[i]
                
                # 4. 座標に変換
                if isinstance(star_list, list) and isinstance(route_indices, list):
                    route_coords = [star_list[idx] for idx in route_indices if idx < len(star_list)]

            except (AttributeError, KeyError, IndexError, TypeError):
                # 'bot_state'がない, キーがない, インデックス範囲外, 型が違うなどの
                # エラーをまとめてキャッチし、何もしない(ルートは空のまま)
                pass
                
        all_routes.append(route_coords)

    # APIのレスポンスに、更新後の最新の星リストとルート情報を追加
    return jsonify(
        coord=final_positions, 
        scores=scores, 
        stars=list(current_stars_set),
        routes=all_routes
    )


if __name__ == "__main__":
    app.run(debug=True)