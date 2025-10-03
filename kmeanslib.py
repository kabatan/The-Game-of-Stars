import random
import pygame
import sys
import time


LOOPNUM=20
def rectdist(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])

def kmeans(star_list, K):
    minx=1000
    maxx=-1
    miny=1000
    maxy=-1
    for x,y in star_list:
        minx=min(minx,x)
        maxx=max(maxx,x)
        miny=min(miny,y)
        maxy=max(maxy,y)

    clustercenter=[]
    clustersize=[0]*K
    for i in range(K):
        clustercenter.append([random.randrange(minx,maxx+1),random.randrange(miny,maxy+1)])
    
    for loop in range(LOOPNUM):
        starowner=[0]*len(star_list)
        clustersum=[[0,0] for i in range(K)]
        clustersize=[0]*K
        for i, starpos in enumerate(star_list):
            mindist=1000
            minj=-1
            for j in range(K):
                dist=rectdist(starpos,clustercenter[j])
                if dist<mindist:
                    mindist=dist
                    minj=j
            starowner[i]=minj
            clustersize[minj]+=1
            clustersum[minj][0]+=starpos[0]
            clustersum[minj][1]+=starpos[1]
        for i in range(K):
            clustercenter[i]=[clustersum[i][0]/clustersize[i], clustersum[i][1]/clustersize[i]]

    clusters=[[] for i in range(K)]
    for i in range(len(star_list)):
        clusters[starowner[i]].append(i)

    global_maxdist=-1
    for i,c in enumerate(clusters):
            mindist=1000
            for idx in c:
                starpos=star_list[idx]
                dist=rectdist(starpos,clustercenter[i])
                if dist<mindist:
                    mindist=dist
            global_maxdist=max(global_maxdist, mindist)

    
    return [clusters, global_maxdist, clustercenter]
            







def kmeansvisualize(clusters, star_list):

    pygame.init()
    cellsize=10
    gridsize=(30,20)
    screen = pygame.display.set_mode((gridsize[0] * cellsize, gridsize[1] * cellsize))

    COLORS = {
    0: (255, 0, 0),       # player 0 = red
    1: (0, 255, 0),       # player 1 = green
    2: (0, 0, 255),       # player 2 = blue
    3: (255, 165, 0),     # player 3 = orange
    4: (128, 0, 128),     # player 4 = purple
    5: (0, 255, 255),     # player 5 = cyan
    6: (255, 20, 146),    # player 6 = pink
}


    
    grid=[]
    for i in range(gridsize[1]):
        grid.append([])
        for j in range(gridsize[0]):
            grid[-1].append(-1)
    
    screen.fill((0, 0, 0))  # background black

    for i,c in enumerate(clusters):
        for j in c:
            grid[star_list[j][0]][star_list[j][1]]=i



    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            color = COLORS.get(cell, (200, 200, 200))  # default gray
            rect = pygame.Rect(x * cellsize, y * cellsize, cellsize, cellsize)

        
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (50, 50, 50), rect, 1)
    pygame.display.flip()
    time.sleep(100)
    

   """  
l=[[0,1],[1,1],[3,1],[10,10]]
c,maxd,m=kmeans(l,2)
kmeansvisualize(c,l)"""
    
        
