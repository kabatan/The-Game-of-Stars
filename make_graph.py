from kmeanslib import kmeans


def make_graph(star_list):
    KRASTERS_NUM = 5
    MAX_KRASTER_DIST = 5
    krasters, maxdist, kraster_centers = kmeans(star_list, KRASTERS_NUM, MAX_KRASTER_DIST)
    while maxdist > MAX_KRASTER_DIST:
        KRASTERS_NUM += 1
        krasters, maxdist, kraster_centers = kmeans(star_list, INITIAL_KRASTERS, MAX_KRASTER_DIST)


    kraster_graph = [[0 for _ in range(len(krasters))] for _ in range(len(krasters))]
    for i in range(len(krasters)):
        r1,c1 = kraster_centers[i]
        for j in range(len(krasters)):
            r2,c2 = kraster_centers[j]
            d = abs(r1-r2)+abs(c1-c2)
            kraster_graph[i][j] = d
            kraster_graph[j][i] = d

    each_kraster_graph = [[[0 for _ in range(len(krasters[i]))] for _ in range(len(krasters[i]))] for i in range(len(krasters))]
    for i in range(len(krasters)):
        for j in range(len(krasters[i])):
            r1,c1 = starlist[krasters[i][j]]
            for k in range(len(krasters[i])):
                r2,c2 = starlist[krasters[i][k]]
                d = abs(r1-r2)+abs(c1-c2)
                each_kraster_graph[i][j][k] = d
                each_kraster_graph[i][k][j] = d

    return kraster_graph, each_kraster_graph
