import random
import heapq
import time

DEFAULT_NB_PLAYERS = 6  # requirement: exactly 6 players by default
ROWS, COLS = 20, 30
STAR_COUNT = 50
PLAYER_ID = 5
star_list = []
star_graph = []
pheromone = []
valid_stars = []

def choose_v(player_idx, v, rest_stars):
    total_pheromone = 0
    for to in star_graph[v]:
        if (rest_stars[to]):
            total_pheromone += pheromone[player_idx][v][to]
    r = random.uniform(0, total_pheromone)
    cumulative = 0
    for to in star_graph[v]:
        if (rest_stars[to]):
            cumulative += pheromone[player_idx][v][to]
            if r <= cumulative:
                return to
    return None

def simulate(ants):
    scores = [0]*len(ants)
    rest_stars = valid_stars
    player_route = [[] for _ in range(len(ants))]
    game_q = []
    for ant_idx in range(len(ants)):
        heapq.heappush(game_q, (0, ant_idx, ant_idx+STAR_COUNT))

    while game_q:
        current = game_q[0][0]
        new = []
        while game_q[0][0] == current:
            _, ant_idx, v_idx = heapq.heappop(game_q)
            new.append((ant_idx, v_idx))
        
        new_pos = {}
        for ant_idx, v_idx in new:
            if (v_idx not in new_pos): new_pos[v_idx] = []
            new_pos[v_idx].append(ant_idx)
            to = choose_v(ant_idx, v_idx, rest_stars)
            if to is not None:
                heapq.heappush(game_q, (current + star_graph[v_idx][to], ant_idx, to))
                player_route[ant_idx].append(to)


        for v_idx in new_pos:
            if (v_idx >= STAR_COUNT): continue
            if (rest_stars[v_idx]):
                chosen_ant=random.randint(0, len(new_pos[v_idx])-1)
                scores[new_pos[v_idx][chosen_ant]]+=1
                rest_stars[v_idx]=False

    return scores, player_route

def init(players, stars):
    star_list = stars
    star_graph = [[0 for _ in range(len(stars)+len(players))] for _ in range(len(stars)+len(players))]
    pheromone = [[[0 for _ in range(len(stars)+len(players))] for _ in range(len(stars)+len(players))] for _ in range(len(players))]
    valid_stars = [True]*len(stars)

    for i in range(len(stars)+len(players)):
        r1,c1 = (stars[i] if i<len(stars) else players[i-len(stars)])
        for j in range(len(stars)+len(players)):
            r2,c2 = (stars[j] if j<len(stars) else players[j-len(stars)])
            d = abs(r1-r2)+abs(c1-c2)
            star_graph[i][j] = d
            star_graph[j][i] = d
            for k in range(len(players)):
                pheromone[k][i][j] = 1.0
                pheromone[k][j][i] = 1.0 

def update_graph(players):
    for i in range(len(players)):
        for j in range(len(star_list)):
            r1,c1 = players[i]
            r2,c2 = star_list[j]
            d = abs(r1-r2)+abs(c1-c2)
            star_graph[i+STAR_COUNT][j] = d
            star_graph[j][i+STAR_COUNT] = d
            if (d == 0):
                valid_stars[j] = False


def move_to_star(pos, star):
    r1, c1 = pos
    r2, c2 = star
    if r1 < r2:
        return (r1 + 1, c1)
    elif r1 > r2:
        return (r1 - 1, c1)
    elif c1 < c2:
        return (r1, c1 + 1)
    elif c1 > c2:
        return (r1, c1 - 1)
    else:
        return (r1, c1)

def move6(players, stars):

    if (star_list == []):
        init(players, stars)
    else:
        update_graph(players)

    end_time = time.perf_counter() + 0.8

    while time.perf_counter() < end_time:
        scores, player_route = simulate(players)
        for i in range(len(players)):
            if (scores[i] > 0):
                for j in range(len(player_route[i])-1):
                    v = player_route[i][j]
                    to = player_route[i][j+1]
                    pheromone[i][v][to] += scores[i]
                    pheromone[i][to][v] += scores[i]

    max_pheromone = -1
    chosen_star = -1
    for i in range(len(stars)):
        if (pheromone[PLAYER_ID][STAR_COUNT+PLAYER_ID][i] > max_pheromone):
            max_pheromone = pheromone[PLAYER_ID][STAR_COUNT+PLAYER_ID][i]
            chosen_star = i

    return move_to_star(players[PLAYER_ID], stars[chosen_star])
