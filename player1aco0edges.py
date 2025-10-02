import heapq
import random
import matplotlib.pyplot as plt

thisplayer=0
playernum=6
rounds=1000
distanceterm=1
decay=0.0001**(1/rounds)
collected=[]

pheromones={}

def rectdis(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])

def pheromoneprob(val):
    #return max(0,val)+0.001
    return 2**val

def pheromoneaccess(v,w, stars):
    if v>w:
        temp=v
        v=w
        w=temp
    if v<playernum:
        return pheromones[str((v,stars[w-playernum]))]
    else:
        return pheromones[str((stars[v-playernum],stars[w-playernum]))]


def pheromoneadd(ownplayer, count, idx):
    if ownplayer:
        return 3*(count)
    else:
        return 0
    
def next(a,b):
    if a[0]>b[0]:
        return (a[0]-1, a[1])
    if a[0]<b[0]:
        return (a[0]+1, a[1])
    if a[1]>b[1]:
        return (a[0], a[1]-1)
    return (a[0], a[1]+1)
    
def move1(players, stars):
    global collected,playernum

    if len(pheromones)==0:
        playernum=len(players)
        for i in range(len(stars)):
            for j in range(i+1,len(stars)):
                pheromones[str((stars[i],stars[j]))]=distanceterm/rectdis(stars[i],stars[j])    
            for j in range(playernum):
                pheromones[str((j,stars[i]))]=distanceterm/rectdis(stars[i],players[j])
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
                    prob[i]=pheromoneprob(pheromoneaccess(v,i+playernum,stars))+distanceterm/adj[v][i]
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
            addlist=[i]+[x+playernum for x in collected[i]]
            for idx,x in enumerate(addlist[:-1]):
                if (x<playernum):
                    pheromones[str((x,stars[addlist[idx+1]-playernum]))]+=(1-decay)*pheromoneadd(i==thisplayer, len(collected),idx)
                else:
                    minval=min(x,addlist[idx+1])-playernum
                    maxval=max(x,addlist[idx+1])-playernum
                    pheromones[str((stars[minval],stars[maxval]))]+=(1-decay)*pheromoneadd(i==thisplayer, len(collected),idx)
        #print(pheromones)
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
            




    