import sys, pygame
from components import Bird,Background
pygame.init()

size = width, height = 600, 800
speed = [1,0]

screen = pygame.display.set_mode(size)


bird=Bird([200,300])
bg=Background()
bg_image = pygame.image.load("background.png")
IMAGE_SIZE = (450, 700)
bg_image = pygame.transform.scale(bg_image, IMAGE_SIZE)
x = 0
screen.blit(bg_image,(x,0))


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
        x -= 1

    if x == -1 * bg_image.get_width():
        x = 0
    screen.blit(bird.image, bird.rect)
    pygame.display.update()