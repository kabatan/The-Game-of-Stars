import heapq
import random
import matplotlib.pyplot as plt

thisplayer=0
playernum=6
rounds=1000
decay=0.01**(1/rounds)
collected=[]

pheromones={}

def rectdis(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])

def pheromoneprob(val):
    #return max(0,val**2)+0.001
    return 2**val


def pheromoneadd(ownplayer, count, idx):
    if ownplayer:
        return 10*(count-idx)
    else:
        return -0.1
    
def next(a,b):
    if a[0]>b[0]:
        return (a[0]-1, a[1])
    if a[0]<b[0]:
        return (a[0]+1, a[1])
    if a[1]>b[1]:
        return (a[0], a[1]-1)
    return (a[0], a[1]+1)
    
def move1(players, stars):
    global collected

    if len(pheromones)==0:
        print("init")
        maxdis=0
        mindis=10**9
        for x in stars:
            maxdis=max(maxdis, rectdis(players[thisplayer], x))
            mindis=min(mindis, rectdis(players[thisplayer], x))
        for x in stars:
            pheromones[str(x)]=3*(1-((rectdis(players[thisplayer],x)-mindis+0.01)/(0.01+maxdis-mindis)))**2
    elif False:

        maxdis=0
        mindis=10**9
        for x in stars:
            maxdis=max(maxdis, rectdis(players[thisplayer], x))
            mindis=min(mindis, rectdis(players[thisplayer], x))
        for x in stars:
            pheromones[str(x)]+=(1-((rectdis(players[thisplayer],x)-mindis+0.01)/(0.01+maxdis-mindis)))**2

    """
    players: list of (r, c) current positions for all 6 players
    stars: list of (r, c) star positions
    return: (r, c) new position for player 2 (max 1 orthogonal step or stay)
    """

    adj=[]
    for i in range(playernum+len(stars)):
        adj.append([])

    for i in range(playernum):
        for j in range(len(stars)):
            adj[i].append(rectdis(players[i],stars[j]))

    for i in range(len(stars)):
        for j in range(i,len(stars)):
            adj[i+playernum].append(rectdis(stars[i],stars[j]))
            if i!=j:
                adj[j+playernum].append(rectdis(stars[i],stars[j]))
    
    successes=0
    maxsucc=0
    data=[]
    for loop in range(rounds):

        visited=[0]*len(stars)
        starnum=len(stars)
        collected=[[] for i in range(playernum)]
        pq=[]
        for i in range(playernum):
            heapq.heappush(pq,(0,-i,i))
        while starnum>0:
            
            dis,player,v=heapq.heappop(pq)
            player=-player
            if (v>=playernum and not visited[v-playernum]):
                starnum-=1
                visited[v-playernum]=True
                collected[player].append(v-playernum)
                if starnum==0:
                    break
            
            if (player==thisplayer):
                prob=[0]*len(stars)
                sum=0
                for i in range(len(stars)):
                    if visited[i] or i==v-playernum:
                        prob[i]=0
                        continue
                    prob[i]=pheromoneprob(pheromones[str(stars[i])])+10/adj[v][i]
                    sum+=prob[i]
                randchoice=random.random()*sum
                choice=-1
            
                while randchoice>=0:
                    choice+=1
                    randchoice-=prob[choice]
                
                
                heapq.heappush(pq, (dis+adj[v][choice], -player, playernum+choice))
            else:
                mindis=10**9
                minidx=-1
                for i in range(len(stars)):
                    if visited[i] or i==v-playernum:
                        continue
                    if adj[v][i]<mindis:
                        mindis=adj[v][i]
                        minidx=i
                heapq.heappush(pq, (dis+adj[v][minidx], -player, playernum+minidx))
                


        for k in list(pheromones.keys()):
            pheromones[k] *= decay
        for i in range(playernum):
            for idx,x in enumerate(collected[i]):
                pheromones[str(stars[x])]+=(1-decay)*pheromoneadd(i==thisplayer, len(collected),idx)
        successes+=len(collected[thisplayer])
        data.append(len(collected[thisplayer]))
        maxsucc=max(maxsucc,len(collected[thisplayer]))
    print(successes/rounds, maxsucc) 
    #plt.plot(data)
    #plt.show()       
    if len(collected[thisplayer])!=0:
        return next(players[thisplayer], stars[collected[thisplayer][0]])
    else:
        return players[thisplayer]#do nothing
            




    