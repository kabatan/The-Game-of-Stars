import random
import heapq
import time
import pickle  
import os    
import math

from var import TIMEOUT, STAR_COUNT,DEFAULT_NB_PLAYERS,ROWS,COLS
from kmeanslib import kmeans

PLAYER_ID = 1
ALPHA = 1.0
BETA = 2.0
RHO = 0.1
Q = 1.0
PHEROMONE_MIN = 1.0
PHEROMONE_MAX = 10000.0
SIMULATETIMES = 10
KRASTERS_NUM = 5
MAX_KRASTER_DIST = 5

''' STATE_FILE = f'player_{PLAYER_ID}_state.pkl' '''

bot_state = {
    "star_list": [],
    "k_graph": [],
    "ks": [],
    "k_dis_ave": [],
    "k_active": [],
    "k_star_rest": [],
    "pheromone": [],
    "valid_stars": [],
    "player_position": []
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


def closest_player(players):
    global bot_state
    closest = [[]]*len(bot_state["star_list"])
    for s_idx in range(len(bot_state["star_list"])):
        if (not bot_state["valid_stars"][s_idx]): continue
        min_d = 1000
        min_player_count = 0
        for p_idx in range(len(players)):
            r1,c1 = players[p_idx]
            r2,c2 = bot_state["star_list"][s_idx]
            d = abs(r1-r2)+abs(c1-c2)
            if (d == min_d):
                closest[s_idx].append(p_idx)
            if (d < min_d):
                min_d = d
                closest[s_idx] = [p_idx]
    return closest

def kraster_reward(players):
    global bot_state
    rewards = [[0 for _ in range(len(bot_state["krasters"]))] for _ in range(len(players))]
    expected_star = [[[] for _ in range(len(bot_state["krasters"]))] for _ in range(len(players))]
    closest = closest_player(players)
    for k_idx in range(len(bot_state["krasters"])):
        for s_idx in bot_state["krasters"][k_idx]:
            if (not bot_state["valid_stars"][s_idx]): continue
            for p_idx in closest[s_idx]:
                rewards[p_idx][k_idx] += 1.0/len(closest[s_idx])
                expected_star[p_idx][k_idx].append(s_idx)
    return rewards

def make_graph(players):
    global bot_state, KRASTERS_NUM, MAX_KRASTER_DIST
    krasters, maxdist, kraster_centers = kmeans(bot_state["star_list"], KRASTERS_NUM, MAX_KRASTER_DIST)
    while maxdist > MAX_KRASTER_DIST:
        KRASTERS_NUM += 1
        krasters, maxdist, kraster_centers = kmeans(bot_state["star_list"], MAX_KRASTER_DIST, MAX_KRASTER_DIST)

    kraster_graph = [[[0 for _ in range(len(krasters[i])+len(players)+1)] for _ in range(len(krasters[i])+len(players)+1)] for i in range(len(krasters))]
    for i in range(len(krasters)):
        for j in range(len(krasters[i])):
            r1,c1 = bot_state["star_list"][krasters[i][j]]
            for k in range(len(krasters[i])):
                r2,c2 = bot_state["star_list"][krasters[i][k]]
                d = abs(r1-r2)+abs(c1-c2)
                kraster_graph[i][j][k] = d
                kraster_graph[i][k][j] = d
            kraster_graph[i][j][len(kraster_graph[i])-1] = 2*MAX_KRASTER_DIST
            kraster_graph[i][len(kraster_graph[i])-1][j] = 2*MAX_KRASTER_DIST

    tmp = [[0 for _ in range(len(krasters)+len(players))] for _ in range(len(krasters)+len(players))]
    for i in range(len(krasters)+len(players)):
        r1,c1 = None,None
        if (i < len(krasters)):
            r1,c1 = kraster_centers[i]
        else:
            r1,c1 = players[i]
        for j in range(len(krasters)+len(players)):
            r2,c2 = None,None
            if (j < len(krasters)):
                r2,c2 = kraster_centers[j]
            else:
                r2,c2 = kraster_centers[j]
            d = abs(r1-r2)+abs(c1-c2)
            tmp[i][j] = d
            tmp[j][i] = d

    kraster_graph.append(tmp)

    return krasters, kraster_graph



def choose_v(player_idx, player_pos, v, rest_stars):
    global bot_state
    total_value = 0.0
    choise = []
    if (player_pos == len(bot_state["ks"])):
        for to in range(len(bot_state["ks"])):
            if k_end[to] != 0:
                tau = bot_state["pheromone"][player_idx][player_pos][v][to]
                dist = bot_state["k_graph"][player_pos][v][to]
                reward = bot_state["k_reward"][player_idx][to]
                value = (tau**ALPHA) * ((Q*reward/(dist+0.01))**BETA)
                choise.append((to, value))
                total_value += value
    else:
        for to in len(bot_state["ks"][player_pos]):
            if (rest_stars[bot_state["ks"][player_pos][to]]):
                tau = bot_state["pheromone"][player_idx][player_pos][v][to]
                dist = bot_state["k_graph"][player_pos][v][to]
                value = (tau**ALPHA) * ((Q/(dist+0.01))**BETA)
                choise.append((to, value))
                total_value += value
            else: 
                choise.append((to, 0.0))

        to = bot_state["ks"][player_pos]+DEFAULT_NB_PLAYERS
        tau = bot_state["pheromone"][player_idx][player_pos][v][to]
        dist = bot_state["k_graph"][player_pos][v][to]
        value = (tau**ALPHA) * ((Q/(dist+0.01))**BETA)
        choise.append((to, value))
        total_value += value 
        
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
    k_rest = bot_state["k_star_rest"][:]
    player_route = [[_ + STAR_COUNT] for _ in range(len(ants))]
    player_pos = bot_state["player_position"][:]
    k_player = [] * len(bot_state["ks"])
    k_now = [None] * len(bot_state["ks"])
    game_q = []
    for ant_idx in range(len(ants)):
        heapq.heappush(game_q, (0, ant_idx, (player_pos[ant_idx], len(bot_state["kraster_graph"][player_pos[ant_idx]]+ant_idx))))

    while game_q:
        current = game_q[0][0]

        for i in range(len(bot_state["ks"])):
            if (len(k_player[i])>0 and current>=k_now[i]+(k_rest[i]*bot_state["k_dis_ave"])/len(k_player[i])):
                for j in k_player[i]:
                    scores[j] += k_rest[i]/len(k_player[i])
                    heapq.heappush(game_q, k_now[i]+math.ceil(k_rest[i]*bot_state["k_dis_ave"])/len(k_player[i]))
                k_rest[i]=0
                
        current = game_q[0][0]
        new = []
        while game_q and game_q[0][0] == current:
            _, ant_idx, v_data = heapq.heappop(game_q)
            new.append((ant_idx, v_data))
        
        new_pos = {}
        for ant_idx, v_data in new:
            if v_data[0] < len(bot_state["ks"]): 
                if v_data not in new_pos: new_pos[v_data] = []
                new_pos[v_data].append(ant_idx)


        for v_data in new_pos:
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

    krasters, kraster_graph = make_graph(players)

    bot_state["krasters"] = krasters
    bot_state["kraster_graph"] = kraster_graph
    bot_state["pheromone"] = [[[[PHEROMONE for _ in range(len(kraster_graph[i]))] for _ in range(len(kraster_graph[i]))] for i in range(len(kraster_graph))] for _ in range(len(players))]
    bot_state["valid_stars"] = [True]*len(stars)
    bot_state["player_position"] = [len(krasters)] * len(players)


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


def pheromone_check():
    global bot_state
    for i in range(len(bot_state["pheromone"])):
        for j in range(len(bot_state["pheromone"][i])):
            for k in range(len(bot_state["pheromone"][i][j])):
                if bot_state["pheromone"][i][j][k] < PHEROMONE_MIN:
                    bot_state["pheromone"][i][j][k] = PHEROMONE_MIN  
                elif bot_state["pheromone"][i][j][k] > PHEROMONE_MAX:
                    bot_state["pheromone"][i][j][k] = PHEROMONE_MAX    


def move2(players, stars):

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
        good_score = [0]*len(players)
        good_route = [[] for _ in range(len(players))]
        for _ in range(SIMULATETIMES):
            scores, player_route = simulate(players)
            for i in range(len(players)):
                if (scores[i] > good_score[i]):
                    good_score[i] = scores[i]
                    good_route[i] = player_route[i]


        for i in range(len(players)):
            if (good_score[i] > 0):
                route = good_route[i]
                for j in range(1, len(route)):
                    v = route[j-1]
                    to = route[j]
                    bot_state["pheromone"][i][v][to] += good_score[i]
                    bot_state["pheromone"][i][to][v] += good_score[i]

            if (good_score[i] > best_score[i]):
                best_score[i] = good_score[i]
                best_route[i] = good_route[i]

        pheromone_check()

    for i in range(len(players)):
        for j in range(len(bot_state["star_list"])+len(players)):
            for k in range(len(bot_state["star_list"])+len(players)):
                bot_state["pheromone"][i][j][k] *= (1.0 - RHO)
    
    '''save_state()'''

    return move_to_star(players[PLAYER_ID], bot_state["star_list"][best_route[PLAYER_ID][1]]) if len(best_route[PLAYER_ID])>1 else players[PLAYER_ID]