import os.path

import neat
import pygame,sys,random,time
pygame.font.init()

Width,Height=500,800

bird_imgs=[pygame.transform.scale2x(pygame.image.load("imgs/bird1.png")),pygame.transform.scale2x(pygame.image.load("imgs/bird2.png")),pygame.transform.scale2x(pygame.image.load("imgs/bird3.png"))]
pipe_img=pygame.transform.scale2x(pygame.image.load("imgs/pipe.png"))
bg_img=pygame.transform.scale2x(pygame.image.load("imgs/bg.png"))
base_img=pygame.transform.scale2x(pygame.image.load("imgs/base.png"))
Stat_font=pygame.font.SysFont("comicsans",50)

class Bird:
	imgs=bird_imgs
	max_rotation=25
	rot_vel=20
	animationtime=5

	def __init__(self,x,y):
		self.x=x
		self.y=y
		self.tilt=0
		self.tick_count=0
		self.vel=0
		self.height=self.y
		self.imgcnt=0
		self.img=self.imgs[0]

	def jump(self):
		self.vel=-10.5
		self.tick_count=0
		self.height=self.y

	def move(self):
		self.tick_count+=1
		d=self.vel*self.tick_count+1.5*self.tick_count**2
		if d>=16:
			d=16
		if d<0:
			d-=2
		self.y=self.y+d
		if d<0 or self.y<self.height+50:
			if self.tilt < self.max_rotation:
				self.tilt=self.max_rotation
		else:
			if self.tilt>-90:
				self.tilt-=self.rot_vel

	def draw(self,win):
		self.imgcnt+=1

		if self.imgcnt < self.animationtime:
			self.img=self.imgs[0]
		elif self.imgcnt < self.animationtime*2:
			self.img=self.imgs[1]
		elif self.imgcnt < self.animationtime*3:
			self.img=self.imgs[2]
		elif self.imgcnt < self.animationtime*4:
			self.img=self.imgs[1]
		elif self.imgcnt == self.animationtime*4+1:
			self.img=self.imgs[0]
			self.imgcnt=0

		if self.tilt<=-80:
			self.img=self.imgs[1]
			self.imgcnt=self.animationtime*2

		rotated_img=pygame.transform.rotate(self.img,self.tilt)
		new_rect=rotated_img.get_rect(center=self.img.get_rect(topleft=(self.x,self.y)).center)
		win.blit(rotated_img,new_rect.topleft)

	def get_mask(self):
		return pygame.mask.from_surface(self.img)


class Pipe:
	GAP=200
	VEL=5

	def __init__(self,x):
		self.x=x
		self.height=0
		self.top=0
		self.bottom=0
		self.pipe_top=pygame.transform.flip(pipe_img,False,True)
		self.pipe_bottom=pipe_img

		self.passed=False
		self.set_height()

	def set_height(self):
		self.height=random.randrange(50,450)
		self.top=self.height-self.pipe_top.get_height()
		self.bottom=self.height+self.GAP

	def move(self):
		self.x-=self.VEL

	def draw(self,win):
		win.blit(self.pipe_top,(self.x,self.top))
		win.blit(self.pipe_bottom,(self.x,self.bottom))

	def collide(self,bird):
		bird_mask=bird.get_mask()
		top_mask=pygame.mask.from_surface(self.pipe_top)
		bottom_mask=pygame.mask.from_surface(self.pipe_bottom)

		top_offset=(self.x-bird.x,self.top-round(bird.y))
		bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

		b_point=bird_mask.overlap(bottom_mask,bottom_offset)
		t_point=bird_mask.overlap(top_mask,top_offset)

		if t_point or b_point:
			return True
		return False


class Base:
	B_Width=base_img.get_width()
	Vel=5
	img=base_img

	def __init__(self,y):
		self.y=y
		self.x1=0
		self.x2=self.B_Width

	def move(self):
		self.x1-=self.Vel
		self.x2-=self.Vel

		if self.x1+self.B_Width<0:
			self.x1=self.x2+self.B_Width

		if self.x2+self.B_Width<0:
			self.x2=self.x1+self.B_Width

	def draw(self,win):
		win.blit(self.img,(self.x1,self.y))
		win.blit(self.img, (self.x2, self.y))


def draw_window(win,birds,pipes,base,score):
	win.blit(bg_img,(0,0))
	for pipe in pipes:
		pipe.draw(win)
	text=Stat_font.render("Score: "+str(score),1,(255,255,255))
	win.blit(text,(Width-10-text.get_width(),10))
	base.draw(win)

	for bird in birds:
		bird.draw(win)
	pygame.display.update()

def main(genomes,config):
	birds=[]
	nets=[]
	ge=[]

	for _,g in genomes:
		net=neat.nn.FeedForwardNetwork.create(g,config)
		nets.append(net)
		birds.append(Bird(230,350))
		g.fitness=0
		ge.append(g)

	base=Base(730)
	pipes=[Pipe(600)]
	win=pygame.display.set_mode((Width,Height))
	clock=pygame.time.Clock()
	score=0

	run=True
	while run:
		clock.tick(30)
		for event in pygame.event.get():
			if event.type==pygame.QUIT:
				run=False
				pygame.quit()
				quit()

		pipe_ind=0
		if len(birds) > 0:
			if len(pipes)>1 and birds[0].x > pipes[0].x+pipes[0].pipe_top.get_width():
				pipe_ind=1
		else:
			run=False
			break

		for x,bird in enumerate(birds):
			bird.move()
			ge[x].fitness+=0.1

			output=nets[x].activate((bird.y,abs(bird.y-pipes[pipe_ind].height),abs(bird.y-pipes[pipe_ind].bottom)))

			if output[0]>0.5:
				bird.jump()


		rem=[]
		add_pipe=False
		for pipe in pipes:
			for x,bird in enumerate(birds):
				if pipe.collide(bird):
					ge[x].fitness-=1
					birds.pop(x)
					nets.pop(x)
					ge.pop(x)
				if not pipe.passed and pipe.x<bird.x:
					pipe.passed=True
					add_pipe=True
			if pipe.x+pipe.pipe_top.get_width()<0:
				rem.append(pipe)
			pipe.move()
		if add_pipe:
			score+=1
			for g in ge:
				g.fitness+=5
			pipes.append(Pipe(600))
		for r in rem:
			pipes.remove(r)
		for x,bird in enumerate(birds):
			if bird.y+base_img.get_height() >=730 or bird.y<0:
				birds.pop(x)
				nets.pop(x)
				ge.pop(x)
		base.move()
		draw_window(win,birds,pipes,base,score)



def run(config_path):
	config=neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,config_path)
	p=neat.Population(config)

	p.add_reporter(neat.StdOutReporter(True))
	stats=neat.StatisticsReporter()
	p.add_reporter(stats)

	winner=p.run(main,50)


if __name__=="__main__":
	local_dir=os.path.dirname(__file__)
	config_path=os.path.join(local_dir,"config-feedforward.txt")
	run(config_path)


