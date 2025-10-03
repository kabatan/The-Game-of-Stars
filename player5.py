# player5.py — Reading (short-term) + Logistic (long-term) + SA(TSP) integration
# 正しい切替: 「敵の最短到達ターン」<= LOOKAHEAD なら短期モデル、そうでなければ長期近似。
# 同時到着は厳密に 1/(同着敵数+1)。短期で誰も来ない範囲で自分が届くなら p=1 を保証。

import math, random, time

from var import TIMEOUT, STAR_COUNT,DEFAULT_NB_PLAYERS,ROWS,COLS

MY_IDX = 4  # player5 (0-indexed)

# -------- Hyperparameters --------
LOOKAHEAD = 4     # 短期読みの手数（敵の到達がこの手数以内なら短期判定）
GAMMA     = 0.97  # 時間割引（早く取るほど価値↑）
ALPHA     = 1.2   # 長期ロジスティックの鋭さ
TOP_M     = 10    # SAで最適化する候補星の数
SA_TIME   = TIMEOUT * 0.8  # SA探索時間（秒）
T0, TMIN, COOL = 2.0, 0.05, 0.996  # 焼きなまし温度スケジュール

# ------------- Utils -------------
def md(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def step_toward(pos, tgt):
    r, c = pos; tr, tc = tgt
    if c < tc:   return (r, c+1)
    if c > tc:   return (r, c-1)
    if r < tr:   return (r+1, c)
    if r > tr:   return (r-1, c)
    return pos

def nearest_star(pos, stars):
    if not stars: return None
    return min(stars, key=lambda s: md(pos, s))

# --------- Reading (short-term) ----------
def simulate_opponents_arrivals(players, stars, lookahead=LOOKAHEAD):
    """
    敵の"同時更新"読みを lookahead 手ぶん行い、
    各星 s に対して { s: (t_min, tie_count) } を返す。
    t_min: その星に最初に到達する敵の手数（1..lookahead）
    tie_count: t_min 同着で到達する敵の人数
    """
    opp_pos = {j: p for j, p in enumerate(players) if j != MY_IDX}
    star_set = set(stars)
    arrival_info = {}  # star -> (t_min, tie_count)

    for step in range(1, lookahead+1):
        if not star_set: break

        stars_list = list(star_set)
        # 1) 各敵が今の最寄り星に1歩「提案」
        proposals = {}
        for j, pos in opp_pos.items():
            tgt = nearest_star(pos, stars_list)
            proposals[j] = step_toward(pos, tgt) if tgt is not None else pos

        # 2) 同時に更新
        for j, np in proposals.items():
            opp_pos[j] = np

        # 3) この手で星に乗った敵を数える（同一星への同時到達もカウント）
        landed_map = {}  # star -> count
        for np in opp_pos.values():
            if np in star_set:
                landed_map[np] = landed_map.get(np, 0) + 1

        # 4) 記録＆星を削除（この手の終わりで消える）
        for s, cnt in landed_map.items():
            arrival_info[s] = (step, cnt)
            star_set.remove(s)

    return arrival_info  # 先読み範囲で誰も到達しない星は含まれない

def static_min_opp_dists(players, stars):
    """長期近似用: 各星の 'マンハッタンでの敵最短到達ターン' を前計算"""
    vals = []
    for s in stars:
        vals.append(min(md(p, s) for j, p in enumerate(players) if j != MY_IDX))
    return vals

# ------- Probability models -------
def prob_short(t_me, t_min, tie_count):
    """
    短期：敵の最短到達 t_min と同着人数 tie_count を用いた厳密確率。
    t_me < t_min -> 1
    t_me = t_min -> 1/(tie_count+1)
    t_me > t_min -> 0
    """
    if t_me < t_min:  return 1.0
    if t_me == t_min: return 1.0 / (tie_count + 1)
    return 0.0

def prob_long(t_me, t_opp_min, alpha=ALPHA):
    """
    長期：差 (t_opp_min - t_me) をロジスティックで近似。
    """
    return 1.0 / (1.0 + math.exp(-alpha * (t_opp_min - t_me)))

def capture_prob(t_me, star, opp_arrivals, t_opp_min, lookahead=LOOKAHEAD):
    """
    期待確率 P(自分が 't_me' ターンで star を取る) を返す。
    短期: その星に対し敵が lookahead 内に到達するなら厳密判定。
          逆に '誰も lookahead 内に来ない' のに 't_me <= lookahead' なら p=1 を保証。
    長期: それ以外はロジスティック近似。
    """
    info = opp_arrivals.get(star)  # (t_min, tie_count) or None
    if info is not None:
        t_min, k = info
        return prob_short(t_me, t_min, k)
    else:
        # 先読み内では敵が来ない
        if t_me <= lookahead:
            return 1.0
        # 先読み外の勝敗は静的最短差で近似
        return prob_long(t_me, t_opp_min)

# --------- Route scoring (TSP EV) ---------
def route_score(order, start_pos, stars, opp_arrivals, t_opp_min_list,
                gamma=GAMMA, lookahead=LOOKAHEAD):
    """
    期待値:  sum_k P_k * gamma^{t_k}
    P_k は capture_prob（短期/長期を正しく切替）
    t_k はスタートからその星までの累積マンハッタン距離
    """
    score, cur, t = 0.0, start_pos, 0
    for idx in order:
        s = stars[idx]
        d = md(cur, s)
        t += d
        p = capture_prob(t, s, opp_arrivals, t_opp_min_list[idx], lookahead)
        score += p * (gamma ** t)
        cur = s
    return score

# ---------- SA Utilities ----------
def greedy_seed(start_pos, stars, opp_arrivals, t_opp_min_list, m=TOP_M):
    """
    初期順序：各星の '即時期待値 p*gamma^d' を評価し上位mを採用。
    その後、距離昇順に並べてSAの初期解にする。
    """
    vals = []
    for i, s in enumerate(stars):
        d = md(start_pos, s)
        p = capture_prob(d, s, opp_arrivals, t_opp_min_list[i])
        v = p * (GAMMA ** d)
        vals.append((v, i))
    vals.sort(reverse=True)
    keep = [i for _, i in vals[:m]]
    keep.sort(key=lambda i: md(start_pos, stars[i]))
    return keep

def sa_optimize(order, start_pos, stars, opp_arrivals, t_opp_min_list,
                time_budget=SA_TIME):
    """
    SAは '順序' だけをいじる。評価は常に期待値スコア。
    """
    if not order:
        return order
    best = order[:]
    best_score = route_score(best, start_pos, stars, opp_arrivals, t_opp_min_list)
    cur, cur_score = best[:], best_score

    T = T0
    t0 = time.perf_counter()
    rng = random.random

    def try_move(seq):
        if len(seq) < 2: return seq
        op = random.randint(0, 2)
        a, b = random.randrange(len(seq)), random.randrange(len(seq))
        if a == b: return seq
        if a > b: a, b = b, a
        new = seq[:]
        if op == 0:  # swap
            new[a], new[b] = new[b], new[a]
        elif op == 1:  # insert
            x = new.pop(b); new.insert(a, x)
        else:  # reverse subseq
            new[a:b+1] = reversed(new[a:b+1])
        return new

    while T > TMIN:
        if time.perf_counter() - t0 > time_budget:
            break
        cand = try_move(cur)
        sc = route_score(cand, start_pos, stars, opp_arrivals, t_opp_min_list)
        dE = sc - cur_score
        if dE >= 0 or rng() < math.exp(dE / max(T, 1e-6)):
            cur, cur_score = cand, sc
            if cur_score > best_score:
                best, best_score = cur[:], cur_score
        T *= COOL
    return best

# --------------- Entry ---------------
def move5(players, stars):
    my_pos = players[MY_IDX]
    if not stars:
        return my_pos

    # 1) 短期：敵の同時更新読み（各星に対する (t_min, tie_count) ）
    opp_arrivals = simulate_opponents_arrivals(players, stars, LOOKAHEAD)

    # 2) 長期：静的な敵最短距離（ロジスティック近似用）
    t_opp_min_list = static_min_opp_dists(players, stars)

    # 3) 初期解
    order0 = greedy_seed(my_pos, stars, opp_arrivals, t_opp_min_list, TOP_M)

    # 4) SAで順序最適化
    order = sa_optimize(order0, my_pos, stars, opp_arrivals, t_opp_min_list, SA_TIME)

    # 5) 最良ルートの先頭星に向けて1歩
    if not order:
        # まれに全部 p≈0 で空になる場合は最寄り星へ
        target = min(stars, key=lambda s: md(my_pos, s))
    else:
        target = stars[order[0]]

    r, c = my_pos; tr, tc = target
    if c < tc:   return (r, c+1)
    if c > tc:   return (r, c-1)
    if r < tr:   return (r+1, c)
    if r > tr:   return (r-1, c)
    return (r, c)
