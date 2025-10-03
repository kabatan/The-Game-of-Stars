import random
import heapq
import time
import pickle  
import os      

from var import TIMEOUT, STAR_COUNT,DEFAULT_NB_PLAYERS,ROWS,COLS

PLAYER_ID = 2
ALPHA = 1.0
BETA = 2.0
RHO = 0.1
Q = 1.0
PHEROMONE_MIN = 0.1

''' STATE_FILE = f'player_{PLAYER_ID}_state.pkl' '''

bot_state = {
    "star_list": [],
    "star_graph": [],
    "pheromone": [],
    "valid_stars": []
}

'''
def save_state():
    with open(STATE_FILE, 'wb') as f:
        pickle.dump(bot_state, f)

def load_state():
    global bot_state
    with open(STATE_FILE, 'rb') as f:
        bot_state = pickle.load(f)
'''

def choose_v(player_idx, v, rest_stars):
    global bot_state
    total_value = 0.0
    choise = []
    for to in range(STAR_COUNT):
        if (rest_stars[to]):
            tau = max(bot_state["pheromone"][player_idx][v][to], PHEROMONE_MIN)
            dist = bot_state["star_graph"][v][to]
            value = (tau**ALPHA) * ((Q/(dist+0.1))**BETA)
            choise.append((to, value))
            total_value += value
        else: 
            choise.append((to, 0.0))
    if total_value == 0.0:
        return None
    
    r = random.uniform(0, total_value)
    cumulative = 0
    for choice_item in choise:
        cumulative += choice_item[1]
        if cumulative >= r:
            return choice_item[0]
    return None

def simulate(ants):
    global bot_state
    scores = [0]*len(ants)
    rest_stars = bot_state["valid_stars"][:]
    player_route = [[_ + STAR_COUNT] for _ in range(len(ants))]
    game_q = []
    for ant_idx in range(len(ants)):
        heapq.heappush(game_q, (0, ant_idx, ant_idx+STAR_COUNT))

    while game_q:
        current = game_q[0][0]
        new = []
        while game_q and game_q[0][0] == current:
            _, ant_idx, v_idx = heapq.heappop(game_q)
            new.append((ant_idx, v_idx))
        
        new_pos = {}
        for ant_idx, v_idx in new:
            if (v_idx not in new_pos): new_pos[v_idx] = []
            new_pos[v_idx].append(ant_idx)


        for v_idx in new_pos:
            if (v_idx >= STAR_COUNT): continue
            if (rest_stars[v_idx]):
                chosen_ant=random.randint(0, len(new_pos[v_idx])-1)
                scores[new_pos[v_idx][chosen_ant]]+=1
                rest_stars[v_idx]=False

        for ant_idx, v_idx in new:
            to = choose_v(ant_idx, v_idx, rest_stars)
            if to is not None:
                heapq.heappush(game_q, (current + bot_state["star_graph"][v_idx][to], ant_idx, to))
                player_route[ant_idx].append(to)

    return scores, player_route

def init(players, stars):
    global bot_state
    bot_state["star_list"] = stars
    bot_state["star_graph"] = [[0 for _ in range(len(stars)+len(players))] for _ in range(len(stars)+len(players))]
    bot_state["pheromone"] = [[[0 for _ in range(len(stars)+len(players))] for _ in range(len(stars)+len(players))] for _ in range(len(players))]
    bot_state["valid_stars"] = [True]*len(stars)

    for i in range(len(stars)+len(players)):
        r1,c1 = (stars[i] if i<len(stars) else players[i-len(stars)])
        for j in range(len(stars)+len(players)):
            r2,c2 = (stars[j] if j<len(stars) else players[j-len(stars)])
            d = abs(r1-r2)+abs(c1-c2)
            bot_state["star_graph"][i][j] = d
            bot_state["star_graph"][j][i] = d
            for k in range(len(players)):
                bot_state["pheromone"][k][i][j] = max(Q/(bot_state["star_graph"][i][j]+0.1), PHEROMONE_MIN)
                bot_state["pheromone"][k][j][i] = max(Q/(bot_state["star_graph"][i][j]+0.1), PHEROMONE_MIN)

def update_graph(players):
    global bot_state

    for i in range(len(players)):
        for j in range(len(bot_state["star_list"])):
            r1,c1 = players[i]
            r2,c2 = bot_state["star_list"][j]
            d = abs(r1-r2)+abs(c1-c2)
            bot_state["star_graph"][i+STAR_COUNT][j] = d
            bot_state["star_graph"][j][i+STAR_COUNT] = d
            if (d == 0):
                bot_state["valid_stars"][j] = False


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
    
def is_new_game(stars):
    for s in stars:
        flag = True
        for t_idx in range(len(bot_state["star_list"])):
            if (bot_state["valid_stars"][t_idx] and s == bot_state["star_list"][t_idx]):
                flag = False
                break
        if (flag): return True
    return False
            

def move3(players, stars):

    global bot_state
    '''
    if (os.path.exists(STATE_FILE)):
        load_state()
    '''

    if (is_new_game(stars)):
        init(players, stars)
    else:
        update_graph(players)

    end_time = time.perf_counter() + TIMEOUT*0.8

    best_score = [0]*len(players)
    best_route = [[] for _ in range(len(players))]
    
    while time.perf_counter() < end_time:
        scores, player_route = simulate(players)
        for i in range(len(players)):
            if (scores[i] > best_score[i]):
                best_score[i] = scores[i]
                best_route[i] = player_route[i]

        for i in range(len(players)):
            if (scores[i] > 0):
                route = player_route[i]
                for j in range(1, len(route)):
                    v = route[j-1]
                    to = route[j]
                    bot_state["pheromone"][i][v][to] += scores[i]
                    bot_state["pheromone"][i][to][v] += scores[i]

    for i in range(len(players)):
        for j in range(len(bot_state["star_list"])+len(players)):
            for k in range(len(bot_state["star_list"])+len(players)):
                bot_state["pheromone"][i][j][k] *= (1.0 - RHO)
    
    '''save_state()'''

    return move_to_star(players[PLAYER_ID], bot_state["star_list"][best_route[PLAYER_ID][1]]) if len(best_route[PLAYER_ID])>1 else players[PLAYER_ID]