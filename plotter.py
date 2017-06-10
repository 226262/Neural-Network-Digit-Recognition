import os
import numpy
import matplotlib.pyplot as plt
import pygame, random

def write_rad(x,y,promien):
    global array
    global width
    if promien>0:
        if (x-promien)>0 and (x+promien)<width and (y-promien)>0 and (y+promien)<width:
            j=0
            for x in range(x-promien,x+promien+1):
                if j<=promien:
                    array[x][y+j]=1
                    array[x][y-j]=1
                    j=j+1
                if j>promien:
                    j=j-1
                    array[x][y+j]
        write_rad(x,y,promien-1)


        

def input_stuff(): 

    width = 300
    height = 300
    screen = pygame.display.set_mode((width,height))
    xMin=width
    xMax=0
    yMin=height
    yMax=0
    array=numpy.full((width,height),0)
    draw_on = False
    last_pos = (0, 0)
    color = (255, 255, 255)
    radius = 3
    edge=0
    isAnythingDrew = False

    def roundline(srf, color, start, end, radius=1):
        global isAnythingDrew
        isAnythingDrew = True
        dx = end[0]-start[0]
        dy = end[1]-start[1]
        distance = max(abs(dx), abs(dy))
        for i in range(distance):
            x = int( start[0]+float(i)/distance*dx)
            y = int( start[1]+float(i)/distance*dy)
            # global xMin,xMax,yMin,yMax
            if x<xMin:
                xMin=x
            if x>xMax:
                xMax=x
            if y<yMin:
                yMin=y
            if y>yMax:
                yMax=y

            write_rad(y,x,1)
            pygame.draw.circle(srf, color, (x, y), radius)

    def cut_and_scale_down(yMin,yMax,xMin,xMax):
        global array
        global edge
        if (yMax-yMin)>=(xMax-xMin):
            edge=yMax-yMin
        else:
            edge=xMax-xMin
        frame=56
        sideFrame=(frame/2)
        tmp_array=numpy.full(((edge+frame),(edge+frame)),0)
        tmp_scaled_array=numpy.full((28,28),0)
        for j in range(int((edge/2)-((xMax-xMin)/2)),int((edge/2)+((xMax-xMin)/2))):
            for i in range(int(sideFrame),int(edge+sideFrame)):
                tmp_array[i][j]=array[yMin+i-int(sideFrame)][xMin+j-int((edge/2)-((xMax-xMin)/2))]
        for i in range(0,(edge+frame-1)):
            for j in range(0,(edge+frame-1)):
                if tmp_array[i][j]==1:
                    tmp_scaled_array[int((i*28)/(edge+frame))][int((j*28)/(edge+frame))]=1
        array=tmp_scaled_array

            

    try:
        while True:
            e = pygame.event.wait()
            if e.type == pygame.QUIT:
                raise StopIteration
            if e.type == pygame.MOUSEBUTTONDOWN:
                # color = (255, 255, 255)
                # pygame.draw.circle(screen, color, e.pos, radius)
                draw_on = True
            if e.type == pygame.MOUSEBUTTONUP:
                draw_on = False
            if e.type == pygame.MOUSEMOTION:
                if draw_on:
                    pygame.draw.circle(screen, color, e.pos, radius)
                    roundline(screen, color, e.pos, last_pos,  radius)
                last_pos = e.pos
            pygame.display.flip()

    except StopIteration:
        pass

    pygame.quit()
    if(isAnythingDrew):
        cut_and_scale_down(yMin,yMax,xMin,xMax)

        return array
        
    else:
        print("You haven't drew anything :c")
        exit()

   
