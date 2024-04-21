#
#This code uses the NEAT library and the Pygame library to create a small game where a bunch of dots try to reach a goal, in this case a red dot
#The dots will slowly create their own neural network using NEAT as the try to find the best way to get past the obstacles and reach the goal
#the player can just sit back and watch as the dots learn how to get to their destination and they compete to be the best species
#for more information on NEAT read: http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf
# January wednseday 29th 2020
#By: Tyler Blinson
import pygame
import random
import sys
import neat
import os
import pickle
import time
pygame.init #initialise pygame
white = (255,255,255)
black = (0,0,0)   #tuples created to define colors
dis_thick = 750
dis_height = 750  #defines the width and height of the screen
display = pygame.display.set_mode ((dis_thick,dis_height)) #creates the display screen
gen = 0      #the number of generation of dots
x1 = random.randint(100,700)
x2 = random.randint(100,700)     #random values used to determine the position of the obstacles
y1 = random.randint(500,600)
y2 = random.randint(200,300)
h=10
# The dot class is the class used to create the dots
#This class allows the dots to move up,down, left and right, it checks if the dots have collided into anything and it allows the dots to be drawn

class Dot:     
	
	def __init__ (self, _x, _y, vel):
		self.x = _x
		self.y = _y
		self.vel = vel

	def moveR(self):
		self.x += self.vel
	
	def moveL(self):
		self.x -= self.vel
	
	def moveU(self):
		self.y -= self.vel
		
	def moveD(self):
		self.y += self.vel
	
	def collide(self,dots,x1, x2,y1,y2, ge):
		for dot in dots:
			if self.x <= 2 or self.x >= 748:
				return True
			if self.y <= 2 or self.y >= 748:
				return True
			if abs(self.y -y1 )< 10 and not (x1 > self.x > (x1-100)):
				return True

			if abs(self.y -y2 )< 10 and not (x2 > self.x > (x2-100)):
				return True
			else:
				return False
	
	def draw(self):
		pygame.draw.circle(display,(black),(self.x,self.y),4)

		#This is the goal class, it represents the dots objective
class Goal :
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def reachGoal(self,dots):     
		for x, dot in enumerate(dots):
			if abs(Dest.x - dot.x) <8 and abs(Dest.y - dot.y) <8: #if the dots reached the goal
				return True
			else:
				return False
	def draw(self):
			pygame.draw.circle(display,(255,0,0),(self.x, self.y),16)  #displays the goal

			#function used to update the display
def Draw(dots,Dest,x1,x2,y1,y2):
        
        Dest.draw()
        for dot in dots:
                dot.draw()
        
        pygame.draw.rect(display,(0,80,0), (x1, y1, 750,h))
        pygame.draw.rect(display,(0,80,0), (x1-100,y1, -700,h))  #draws the obstacles
        pygame.draw.rect(display,(0,80,0), (x2, y2, 750,h))
        pygame.draw.rect(display,(0,80,0), (x2-100,y2, -800,h))
        pygame.display.update()       #updates the display
	
def eval_genomes(genomes, config):  #the main loop
	moves = 0  #the amount of times a dot has moved
	Dest = Goal(375, 50)  #decides the location of the goal
	global display, gen 
	gen+=1            #tells how many generation passed
	nets = []   #creates the arrays used for NEAT
	ge = []
	dots = []
	x1 = random.randint(100,700)
	x2 = random.randint(100,700)
	y1 = random.randint(500,600)
	y2 = random.randint(200,300)
	for _, g in genomes:
		g.fitness = 0    #makes the dots fitness equal to 0
		net = neat.nn.FeedForwardNetwork.create(g, config)
		nets.append(net)    #creates an array of the config file for the dots to use
		d = Dot(50,650,3)   #creates an object of the dot class
		dots.append(d)      #fills up the dots array with the dot object
		ge.append(g)         #creates an array of genomes for the dots to use
	running = True
	while running and len(dots) > 0 and moves < 650:  #while theres dots alive and less than 650 moves have been made
		pygame.time.delay(5)   #delay of 5 milliseconds
		for event in pygame.event.get():   #if the x is pressed, end the generation
			if event.type == pygame.QUIT:
				running = False

		for x, dot in enumerate(dots):
			#for every dot in the dots array, give them the position of themselves, the position of the goal, and the position of the obstacles
			#then based on their neural network, it decides if they shoul go up down, lef or right
			output = nets[dots.index(dot)].activate((dot.x, dot.y, (Dest.x - dot.x), (Dest.y - dot.y), (x1 -dot.x), ((x1-100)-dot.x), (x2-dot.x) ,((x2-100)-dot.x), (y1-dot.y), (y2 - dot.y)))
			
			if output[0] > 0.0 :
				dot.moveU()
				ge[dots.index(dot)].fitness += 0.01
			
			if output[1] > 0.0 :
				dot.moveR()
				ge[dots.index(dot)].fitness += 0.01
			
			if output[2] > 0.0:
				dot.moveL()
				ge[dots.index(dot)].fitness += 0.01
		
			if output[3] > 0.0:
				dot.moveD()
				ge[dots.index(dot)].fitness += 0.01
			
		for dot in dots:              
			if dot.collide(dots,x1,x2,y1,y2 ,ge):   #checks every dot to see if this move made them collide with something
				if (dot.y > y1):               #if theyre under the first obstacle
					ge[dots.index(dot)].fitness -=30   #remove 30 fitness
				if (dot.y < y1 and dot.y >y2):   #if they got over the first obstacle
					ge[dots.index(dot)].fitness += 10  #gives 10 points
					ge[dots.index(dot)].fitness += 40 - ((abs(dot.x - (x2-50)) + abs(dot.y - y2)) * 0.05)  #give them extrs points based off of how close they are from the opening of the second obstacle
				if dot.y < y2 :                   #if they got over both obstacles
					ge[dots.index(dot)].fitness += 20   #give them 40 points
					ge[dots.index(dot)].fitness += 70- (abs(dot.x - Dest.x) + abs(dot.y - Dest.y)) * 0.05  #give them extra points based oh their proximity to the goal
				ge[dots.index(dot)].fitness -= 20  #removes 20 points for hitting an obstacle
				nets.pop(dots.index(dot))
				ge.pop(dots.index(dot))        #removes the dot, their genomes and config file from the array
				dots.pop(dots.index(dot))
			elif abs(Dest.x - dot.x) <13 and abs(Dest.y - dot.y) <13:   #if the dot reached the destination
				ge[dots.index(dot)].fitness += 185   #gives them 185 points
				ge[dots.index(dot)].fitness -= (moves * 0.01)  #removes points based on the amount of moves made
				nets.pop(dots.index(dot))
				ge.pop(dots.index(dot))      #removes the dot, its genomes and the config file from the array
				dots.pop(dots.index(dot))
		display.fill(white) #fills the display in white
		Draw(dots, Dest,x1,x2,y1,y2)  #redraws the dots, destination and obstacles
		moves += 1  #add one to the move counter
	for dot in dots:
		if (dot.y > y1):   #if the dot didnt pass the first obstacle
			ge[dots.index(dot)].fitness -=30 #remove 30 points
		if (dot.y < y1 and dot.y >y2):  #if the dot passed the second obstacle
			ge[dots.index(dot)].fitness += 10  #give them 10 points
			ge[dots.index(dot)].fitness += 40 - ((abs(dot.x - (x2-50)) + abs(dot.y - y2)) * 0.05)  #give them extrs points based off of how close they are from the opening of the second obstacle
		if dot.y < y2 -10 :   #if the dot got passed both obstacles
			ge[dots.index(dot)].fitness += 20   #give them 70 points
			ge[dots.index(dot)].fitness += 70- (abs(dot.x - Dest.x) + abs(dot.y - Dest.y)) * 0.05  #give them points based off of how close they are to the destination

def run(config_file):    #function used to locate the config file and calls the "main" loop
	config = neat.config.Config(neat.DefaultGenome,neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
	p = neat.Population(config)
	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()    #gives the stats on the dots fitness
	p.add_reporter(stats) 
	winner = p.run(eval_genomes,999999)    #definne how many generations it runs for
if __name__ == '__main__':
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, "config-feedforward.txt")
	run(config_path)
