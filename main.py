import player1aco0edges as player1
import player2
import player3
import player4
import player5
import player6

from random import randrange
import pygame
import sys



def distance(p1,p2):
    return (abs(p1[0]-p2[0])+abs(p1[1]-p2[1]))

gridsize=(30,20)
starnum=50
mindis=4
maxmoves=80
cellsize=10
playerpos=[[0,0],[15,0],[29,0],[0,19],[14,19],[29,19]]
starpos=[]
for i in range(starnum):
    starpos.append([0,0])
scores=[0]*6

grid=[]#-1 blanc, -2 star, 0-6 players
for i in range(gridsize[1]):
    grid.append([])
    for j in range(gridsize[0]):
        grid[-1].append(-1)


COLORS = {
    -1: (220, 220, 220),  # blank = light gray
    -2: (255, 215, 0),    # star = gold
    0: (255, 0, 0),       # player 0 = red
    1: (0, 255, 0),       # player 1 = green
    2: (0, 0, 255),       # player 2 = blue
    3: (255, 165, 0),     # player 3 = orange
    4: (128, 0, 128),     # player 4 = purple
    5: (0, 255, 255),     # player 5 = cyan
    6: (255, 20, 146),    # player 6 = pink
}

def show(screen, grid):
    """Draw the grid with pygame."""
    screen.fill((0, 0, 0))  # background black

 

    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            color = COLORS.get(cell, (200, 200, 200))  # default gray
            rect = pygame.Rect(x * cellsize, y * cellsize, cellsize, cellsize)

           
            pygame.draw.rect(screen, color, rect)


            # draw grid lines
            pygame.draw.rect(screen, (50, 50, 50), rect, 1)

    if len(player1.collected)!=0:
        points = [(playerpos[0][0]*cellsize+cellsize // 2,playerpos[0][1]*cellsize+cellsize // 2)]
        for idx in player1.collected[0]:
            sx, sy = starpos[idx]
            px = sx * cellsize + cellsize // 2
            py = sy * cellsize + cellsize // 2
            points.append((px, py))

        # draw connecting line if at least 2 stars are collected
        if len(points) >= 2:
            pygame.draw.lines(screen, (0, 0, 0), False, points, 3)
            print("drawn")

    pygame.display.flip()




for i in range(6):
    grid[playerpos[i][1]][playerpos[i][0]]=i

for i in range(starnum):
    pos=[-1,-1]
    while True:
        pos=[randrange(0,gridsize[0]),randrange(0,gridsize[1])]
        dis=1000
        for j in range(6):
            dis=min(dis,distance(pos,playerpos[j]))
        if dis>=mindis and pos not in starpos:
            break
    starpos[i]=pos
    print(pos)
    grid[pos[1]][pos[0]]=-2



pygame.init()
screen = pygame.display.set_mode((gridsize[0] * cellsize, gridsize[1] * cellsize))
pygame.display.set_caption("Grid Field")
clock = pygame.time.Clock()

moveloop=0
while len(starpos) > 0 and moveloop<maxmoves:
    moveloop+=1
    print(moveloop)
    # Handle quit events so window doesn’t freeze
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # --- Build grid for display ---
    
    

    

    # --- Player moves ---
    wishes = {}
    for i in range(6):
        wish = eval(f"player{i+1}.move{i+1}(playerpos,starpos)")  # returns (x,y)
        wishes.setdefault(str(wish), [])
        wishes[str(wish)].append(i)

    # Show field
    show(screen, grid)
    
    for k, v in wishes.items():
        if len(v) > 1:
            continue
        if grid[playerpos[v[0]][1]][playerpos[v[0]][0]]==v[0]:
            grid[playerpos[v[0]][1]][playerpos[v[0]][0]]=-1
        playerpos[v[0]] = eval(k)
        grid[playerpos[v[0]][1]][playerpos[v[0]][0]]=v[0]
        if list(eval(k)) in starpos:
            scores[v[0]] += 1
            starpos.remove(list(eval(k)))

    #pygame.image.save(screen, f"frames/frame_{moveloop:04d}.png")
    clock.tick(5)  # slow down loop so we can see changes (3 FPS)

print("Final scores:", scores)
pygame.quit()
sys.exit()
