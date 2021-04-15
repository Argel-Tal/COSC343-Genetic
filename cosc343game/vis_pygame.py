import pygame
import numpy as np
import sys

class visualiser:

   def __init__(self, speed, gridSize, resolution=(720,480), playerStrings=None):
      pygame.init()

      self.gridSize = gridSize
      self.playerStrings = playerStrings
      self.left_frame = 100

      self.width, self.height = resolution
      self.WHITE = (255, 255, 255)
      self.BLACK = 0, 0, 0

      if speed == "normal":
         self.frameTurns = 20
         self.nSteps = 10
      elif speed == "fast":
         self.frameTurns = 1
         self.nSteps = 5
      elif speed == "slow":
         self.frameTurns = 40
         self.nSteps = 10

      self.screen = pygame.display.set_mode(resolution)

      self.unit = int(np.min([self.width - self.left_frame, self.height]) / self.gridSize)

      self.im_creatures = [pygame.image.load('images/creature_blue.png'),
                      pygame.image.load('images/creature_red.png')
                      ]

      self.im_food = pygame.image.load('images/strawberry-green.png')
      self.strawb_size = int(self.unit*0.7)
      self.im_food = pygame.transform.scale(self.im_food, (self.strawb_size, self.strawb_size))

      self.im_creatures_scaled = [list(), list()]
      self.im_creatures_scales = list()

      max_size=8
      unit_size=4

         # self.im_creatures[i] = pygame.transform.scale(self.im_creatures[i], (self.unit, self.unit))
      for j in range(1,max_size+1):
        area = self.unit*self.unit*j/unit_size
        side = int(np.sqrt(area))
        if side < 1:
            side = 1

        self.im_creatures_scales.append(side)

        for i in range(len(self.im_creatures)):
            self.im_creatures_scaled[i].append(pygame.transform.scale(self.im_creatures[i], (side, side)))

      self.font = pygame.font.Font("arial.ttf", 14)
      self.reset()

   def __del__(self):
      pygame.display.quit()
      pygame.quit()

   def reset(self):
      self.prev_creature_state = None

   def show(self, creature_state, food_array,wall_array,game=None,turn=0,titleStr=None):
      if titleStr is None:
          caption = ''
      else:
          caption = titleStr + ', '

      if game is not None:
         if isinstance(game,str):
            caption += 'Game %s ' % game
         else:
             caption += 'Game %d' % game
             if turn>0:
                caption += ", "

      if turn>0:
          caption += 'Turn %d' % (turn)

      pygame.display.set_caption(caption)


      I = np.where(creature_state[:,2]==1)[0]
      nCreatures2 = int(np.sum(creature_state[I,3]))
      nCreatures1 = len(I)-nCreatures2


      for k in range(self.nSteps):

         for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

         self.screen.fill(self.WHITE)

         # render text
         if self.playerStrings is not None:
             label = self.font.render( self.playerStrings[0], 1, (33, 79, 255))
             self.screen.blit(label, (10, 10))
             label = self.font.render( "Creatures: %d" % nCreatures1, 1, (33, 79, 255))
             self.screen.blit(label, (10, 30))

             label = self.font.render(self.playerStrings[1], 1, (230, 42, 55))
             self.screen.blit(label, (self.left_frame + (self.gridSize * self.unit)+10, 10))
             label = self.font.render( "Creatures: %d" % nCreatures2, 1, (230, 42, 55))
             self.screen.blit(label, (self.left_frame + (self.gridSize * self.unit)+10, 30))

         for i in range(self.gridSize + 1):
            pygame.draw.line(self.screen, self.BLACK, [self.left_frame, i * self.unit],
                             [self.left_frame + (self.gridSize * self.unit), i * self.unit])
            pygame.draw.line(self.screen, self.BLACK, [self.left_frame + (i * self.unit), 0],
                             [self.left_frame + (i * self.unit), self.gridSize * self.unit])

         for (x,y) in wall_array:
            pygame.draw.rect(self.screen,self.BLACK, (self.left_frame + (x * self.unit), y*self.unit, self.unit, self.unit))

         obj_im = self.im_food
         for (x,y) in food_array:
            im_offset = 0 #int((self.unit - self.im_food) / 2)

            obj_loc = pygame.Rect(self.left_frame + x * self.unit + im_offset,
                                           y * self.unit + im_offset, self.unit,
                                           self.unit)
            self.screen.blit(obj_im, obj_loc)

         stepDiff = 1.0 / float(self.nSteps)
         halfSteps = int(np.floor(self.nSteps / 2))

         if self.prev_creature_state is None:
            self.prev_creature_state = creature_state

         for i in range(len(creature_state)):
            (x,y,a,p,_) = creature_state[i]
            if a==1:
               (xprev,yprev,_,_,s) = self.prev_creature_state[i]

               xshift = xprev-x
               if np.abs(xshift)<=1:
                   xdiff = (x - xprev) * k * stepDiff
               elif k <= halfSteps:
                   xdiff = np.sign(xshift) * k * stepDiff
               else:
                   xdiff = -np.sign(xshift) * k * stepDiff
                   xprev = x

               yshift = yprev - y
               if np.abs(yshift) <= 1:
                   ydiff = (y - yprev) * k * stepDiff
               elif k <= halfSteps:
                   ydiff = np.sign(yshift) * k * stepDiff
               else:
                   ydiff = -np.sign(yshift) * k * stepDiff
                   yprev = y

               s = int(s)
               if s >= len(self.im_creatures_scaled[p]):
                   s = len(self.im_creatures_scaled[p])-1

               obj_im = self.im_creatures_scaled[p][s]
               im_offset = int((self.unit-self.im_creatures_scales[s])/2)

               obj_loc = pygame.Rect(self.left_frame + (xprev + xdiff) * self.unit + im_offset, (yprev + ydiff) * self.unit + im_offset, self.unit,
                                               self.unit)
               self.screen.blit(obj_im, obj_loc)

         pygame.display.flip()
         pygame.time.delay(self.frameTurns)

      self.prev_creature_state = np.copy(creature_state)
