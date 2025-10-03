import random


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
                print(starpos)
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
            

        

