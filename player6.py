WEIGHTS = [1.0, 0.8, 0.6, 0.4, 0.2]

def _score(p, stars):
    if not stars: return 0.0
    x, y = p
    d = sorted(abs(x-sx) + abs(y-sy) for sx, sy in stars)[:5]
    return sum(w * (1/(1+di)) for w, di in zip(WEIGHTS, d))

def move6(players, stars):
    x, y = players[5]
    cand = [(x+1,y), (x-1,y), (x,y-1), (x,y+1)]  # R, L, U, D
    return max(cand, key=lambda q: _score(q, stars))