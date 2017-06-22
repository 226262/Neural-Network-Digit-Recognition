import os
import numpy
import matplotlib.pyplot as plt
import pygame, random


class Plotter:
        
    width=300
    height=width 
    array=numpy.full((width,height),0)
    xMin=width
    xMax=0
    yMin=height
    yMax=0
    edge=0
    isAnythingDrew = False

    def write_rad(self,x,y,promien):
        
        if promien>0:
            if (x-promien)>0 and (x+promien)<self.width and (y-promien)>0 and (y+promien)<self.width:
                j=0
                for x in range(x-promien,x+promien+1):
                    if j<=promien:
                        self.array[x][y+j]=1
                        self.array[x][y-j]=1
                        j=j+1
                    if j>promien:
                        j=j-1
                        self.array[x][y+j]
            self.write_rad(x,y,promien-1)


    def cut_and_scale_down(self):
       
        if (self.yMax-self.yMin)>=(self.xMax-self.xMin):
            edge=self.yMax-self.yMin
        else:
            edge=self.xMax-self.xMin
        frame=56
        sideFrame=(frame/2)
        tmp_array=numpy.full(((edge+frame),(edge+frame)),0)
        tmp_scaled_array=numpy.full((28,28),0)
        for j in range(int((((edge+frame)/2)-(self.xMax-self.xMin)/2)),int(((edge+frame)/2)+((self.xMax-self.xMin)/2))):
            for i in range(int(sideFrame),int(edge+sideFrame)):
                tmp_array[i][j]=self.array[self.yMin+i-int(sideFrame)][self.xMin+j-int(((edge+frame)/2)-((self.xMax-self.xMin)/2))]
        for i in range(0,(edge+frame-1)):
            for j in range(0,(edge+frame-1)):
                if tmp_array[i][j]==1:
                    tmp_scaled_array[int((i*28)/(edge+frame))][int((j*28)/(edge+frame))]=1
        self.array=tmp_scaled_array
        # print(self.array)
        
    def input_stuff(self):

         

        screen = pygame.display.set_mode((self.width,self.height))
        draw_on = False
        last_pos = (0, 0)
        color = (255, 255, 255)
        radius = 3
        

        def roundline(srf, color, start, end, radius=1):
           
            self.isAnythingDrew = True
            dx = end[0]-start[0]
            dy = end[1]-start[1]
            distance = max(abs(dx), abs(dy))
            for i in range(distance):
                x = int( start[0]+float(i)/distance*dx)
                y = int( start[1]+float(i)/distance*dy)
                if x<self.xMin:
                    self.xMin=x
                if x>self.xMax:
                    self.xMax=x
                if y<self.yMin:
                    self.yMin=y
                if y>self.yMax:
                    self.yMax=y

                self.write_rad(y,x,2)
                pygame.draw.circle(srf, color, (x, y), radius)

                

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
        if(self.isAnythingDrew):
            self.cut_and_scale_down()

            return self.array
            
        else:
            print("You haven't drew anything :c")
            exit()

    def flush(self):
        self.array=numpy.full((self.width,self.height),0)
        self.xMin=self.width
        self.xMax=0
        self.yMin=self.height
        self.yMax=0
        self.edge=0
        self.isAnythingDrew = False


   
