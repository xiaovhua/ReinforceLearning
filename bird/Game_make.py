# Used to paint the Game with a state of bird
import pygame
import os

class game_make():
    def __init__(self, path=os.path.split(os.path.realpath(__file__))[0], screen_size=(800, 493)):
        # load the picture of bird, bride and sky, initial the screen's size
        self.path = path
        self.bird_path = self.path + '/resources/bird.png'
        self.obstacle_path = self.path + '/resources/obstacle.png'
        self.background_path = self.path + '/resources/background.png'
        self.screen_size = screen_size
        self.viewer = None
        self.state = [0, 0]

    def assign_state(self, state):
        # decide where to paint the bird
        self.state = state

    def init_screen(self):
        # create the viewer object
        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode(self.screen_size, 0, 32)

    def load_picture(self):
        # create the objects by using the pictures
        self.bird = pygame.image.load( self.bird_path ).convert_alpha()
        self.obstacle = pygame.image.load( self.obstacle_path ).convert_alpha()
        self.background = pygame.image.load( self.background_path ).convert_alpha()

    def flash_background(self):
        # paint the whole game
        self.init_screen()
        self.load_picture()
        self.viewer.blit(self.background, (0, 0))
        self.viewer.blit(self.bird, (self.state[1] * 40, self.state[0] * 29))
        self.viewer.blit(self.bird, (self.screen_size[0]-40, 0))
        # calculate the site of bride
        for i in range(9):
            self.viewer.blit(self.obstacle, (280, i * 29))
            self.viewer.blit(self.obstacle, (280, i * 29 + 9))
            self.viewer.blit(self.obstacle, (560, self.screen_size[1] - 29 - i * 29))
            self.viewer.blit(self.obstacle, (560, self.screen_size[1] - 29 - i * 29 + 20))
        for i in range(4):
            self.viewer.blit(self.obstacle, (280, self.screen_size[1] - 29 - i * 29))
            self.viewer.blit(self.obstacle, (280, self.screen_size[1] - 29 - i * 29 + 20))
            self.viewer.blit(self.obstacle, (560, i * 29))
            self.viewer.blit(self.obstacle, (560, i * 29 + 9))
        # paint
        pygame.display.update()

