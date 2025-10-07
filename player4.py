import random
import heapq
import time
import pickle  
import os     
import math 

from var import TIMEOUT, STAR_COUNT,DEFAULT_NB_PLAYERS,ROWS,COLS

PLAYER_ID = 3
ALPHA = 1.0
BETA = 2.0
GAMMA = 1.0
RHO = 0.05
Q = 1.0
SIMULATETIMES = 50
GLOBALBESTBONUS = 5

''' STATE_FILE = f'player_{PLAYER_ID}_state.pkl' '''

bot_state = {
    "star_list": [],
    "star_graph": [],
    "pheromone": [],
    "valid_stars": [],
    "valid_stars_num": 0,
    "global_best_score": [0]*DEFAULT_NB_PLAYERS,
    "global_best_route": [[] for _ in range(DEFAULT_NB_PLAYERS)]
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


def pheromone_max(player_idx):
    global bot_state
    return (1.0/RHO) * (bot_state["global_best_score"][player_idx]**GAMMA)

def pheromone_min(player_idx):
    global bot_state
    tmp = max(3, bot_state["valid_stars_num"])
    return (1.0-math.pow(0.05, 1.0/tmp))*pheromone_max(player_idx)/((tmp/2.0-1.0)*math.pow(0.05, 1.0/tmp))


def choose_v(player_idx, v, rest_stars):
    global bot_state
    total_value = 0.0
    choise = []
    for to in range(STAR_COUNT):
        if (rest_stars[to]):
            tau = bot_state["pheromone"][player_idx][v][to]
            dist = bot_state["star_graph"][v][to]
            value = (tau**ALPHA) * ((Q/(dist+0.001))**BETA)
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
    player_route = [[] for _ in range(len(ants))]
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

def simulate_greedy(ants):
    global bot_state
    scores = [0]*len(ants)
    rest_stars = bot_state["valid_stars"][:]
    player_route = [[] for _ in range(len(ants))]
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
            to = None
            min_dist = float('inf')
            for star in range(STAR_COUNT):
                if rest_stars[star] and bot_state["star_graph"][v_idx][star] < min_dist:
                    min_dist = bot_state["star_graph"][v_idx][star]
                    to = star
            if to is not None:
                heapq.heappush(game_q, (current + bot_state["star_graph"][v_idx][to], ant_idx, to))
                player_route[ant_idx].append(to)

    return scores, player_route

def simulate_with_global_best(ants, player_idx, suggested_routes):
    global bot_state
    scores = [0]*len(ants)
    rest_stars = bot_state["valid_stars"][:]
    player_routes = [[] for i in range(len(ants))]
    game_q = []
    for ant_idx in range(len(ants)):
        heapq.heappush(game_q, (0, ant_idx, ant_idx+STAR_COUNT))
        if (player_idx == ant_idx and suggested_routes):
            player_routes[ant_idx] = suggested_routes[:]
        elif bot_state["global_best_route"][ant_idx]:
            player_routes[ant_idx] = bot_state["global_best_route"][ant_idx][:]

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
            to = None
            if player_routes[ant_idx]:
                to = player_routes[ant_idx][0]
                player_routes[ant_idx] = player_routes[ant_idx][1:]
            if to is not None:
                heapq.heappush(game_q, (current + bot_state["star_graph"][v_idx][to], ant_idx, to))

    return scores

def init(players, stars):
    global bot_state
    bot_state["star_list"] = stars
    bot_state["star_graph"] = [[0 for _ in range(len(stars)+len(players))] for _ in range(len(stars)+len(players))]
    bot_state["pheromone"] = [[[0 for _ in range(len(stars)+len(players))] for _ in range(len(stars)+len(players))] for _ in range(len(players))]
    bot_state["valid_stars"] = [True]*len(stars)
    bot_state["valid_stars_num"] = STAR_COUNT

    for i in range(len(stars)+len(players)):
        r1,c1 = (stars[i] if i<len(stars) else players[i-len(stars)])
        for j in range(len(stars)+len(players)):
            r2,c2 = (stars[j] if j<len(stars) else players[j-len(stars)])
            d = abs(r1-r2)+abs(c1-c2)
            bot_state["star_graph"][i][j] = d
            bot_state["star_graph"][j][i] = d

    bot_state["global_best_score"], bot_state["global_best_route"] = simulate_greedy(players)

    for i in range(len(players)):
        for j in range(len(stars)+len(players)):
            for k in range(len(stars)+len(players)):
                bot_state["pheromone"][i][j][k] = pheromone_min(i)
                bot_state["pheromone"][i][k][j] = pheromone_min(i)


def update_graph(players):
    global bot_state

    for i in range(len(players)):
        for j in range(len(bot_state["star_list"])):
            r1,c1 = players[i]
            r2,c2 = bot_state["star_list"][j]
            d = abs(r1-r2)+abs(c1-c2)
            bot_state["star_graph"][i+STAR_COUNT][j] = d
            bot_state["star_graph"][j][i+STAR_COUNT] = d
            if (d == 0 and bot_state["valid_stars"][j]):
                bot_state["valid_stars"][j] = False
                bot_state["valid_stars_num"] -= 1

    for i in range(len(players)):
        route = bot_state["global_best_route"][i]
        if (route and players[i] == bot_state["star_list"][bot_state["global_best_route"][i][0]]):
            bot_state["global_best_score"][i] -= 1
            bot_state["global_best_route"][i] = bot_state["global_best_route"][i][1:]

    bot_state["global_best_score"] = simulate_with_global_best(players, 0, bot_state["global_best_route"][0])
    


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


def pheromone_check():
    global bot_state
    for i in range(len(bot_state["pheromone"])):
        for j in range(len(bot_state["pheromone"][i])):
            for k in range(len(bot_state["pheromone"][i][j])):
                if bot_state["pheromone"][i][j][k] < pheromone_min(i):
                    bot_state["pheromone"][i][j][k] = pheromone_min(i)  
                elif bot_state["pheromone"][i][j][k] > pheromone_max(i):
                    bot_state["pheromone"][i][j][k] = pheromone_max(i)  


def move4(players, stars):

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
    t=0
    
    while time.perf_counter() < end_time:
        good_score = [0]*len(players)
        good_route = [[] for _ in range(len(players))]
        for _ in range(SIMULATETIMES):
            scores, player_route = simulate(players)
            for i in range(len(players)):
                if (scores[i] > good_score[i]):
                    good_score[i] = scores[i]
                    good_route[i] = player_route[i]
            t += 1


        for i in range(len(players)):

            test_score = simulate_with_global_best(players, i, good_route[i])

            I = 1.0
            is_best = False

            if (test_score[i] > bot_state["global_best_score"][i]):
                I = (test_score[i]/(bot_state["global_best_score"][i]+0.01))
                bot_state["global_best_score"][i] = test_score[i]
                bot_state["global_best_route"][i] = good_route[i]
                is_best = True
            

            if (test_score[i] > 0):
                route = good_route[i]
                for j in range(1, len(route)):
                    v = route[j-1]
                    to = route[j]
                    if (is_best):
                        bot_state["pheromone"][i][v][to] += I * (test_score[i]**GAMMA) * (GLOBALBESTBONUS ** ((1.0-(j-1)/(2*len(route)))))
                        bot_state["pheromone"][i][to][v] += I * (test_score[i]**GAMMA) * (GLOBALBESTBONUS ** ((1.0-(j-1)/(2*len(route)))))
                    else:
                        bot_state["pheromone"][i][v][to] += I * (test_score[i]**GAMMA)
                        bot_state["pheromone"][i][to][v] += I * (test_score[i]**GAMMA)



        pheromone_check()

    for i in range(len(players)):
        for j in range(len(bot_state["star_list"])+len(players)):
            for k in range(len(bot_state["star_list"])+len(players)):
                bot_state["pheromone"][i][j][k] *= (1.0 - RHO)
    
    '''save_state()'''
    '''print(f"player{PLAYER_ID} simulate times: {t}")'''

    return move_to_star(players[PLAYER_ID], bot_state["star_list"][bot_state["global_best_route"][PLAYER_ID][0]]) if len(bot_state["global_best_route"][PLAYER_ID])>=1 else players[PLAYER_ID]