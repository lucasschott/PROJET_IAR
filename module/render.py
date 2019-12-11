import numpy as np
import time
import cma
import argparse
from functools import partial
import matplotlib.pyplot as plt
import pygame

import multiprocessing as mp
from build import pylib

input = 12
output = 4

LAYER = 12
PRED_NETWORK_SIZE = input*LAYER+LAYER + LAYER*output+output
PREY_NETWORK_SIZE = (input*2)*LAYER+LAYER + LAYER*output+output

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

def load_prey(path):
    if path == "random":
        return list(np.random.rand(PREY_NETWORK_SIZE))
    return list(np.load(path))

def load_pred(path):
    if path == "random":
        return list(np.random.rand(PRED_NETWORK_SIZE))
    return list(np.load(path))

parser = argparse.ArgumentParser()

parser.add_argument('--input', default=12, type=int)
parser.add_argument('--fps', default=60, type=int)
parser.add_argument('--output', default=4, type=int)
parser.add_argument('--num_preys', default=50, type=int)
parser.add_argument('--num_predators', default=1, type=int)
parser.add_argument('--eat_distance', default=9, type=int)
parser.add_argument('--confusion', dest='confusion', action='store_true')
parser.add_argument('--no-confusion', dest='confusion', action='store_false')
parser.set_defaults(confusion=True)
parser.add_argument('--timesteps', default=2000, type=int)
parser.add_argument('--pred', default='random', type=str)
parser.add_argument('--prey', default='random', type=str)
parser.add_argument('--env_x', default=512, type=int)
parser.add_argument('--env_y', default=512, type=int)

args = parser.parse_args()

pred = load_pred(args.pred)
prey = load_prey(args.prey)

print(args.confusion)
s = pylib.Simulation(args.input, args.output, args.num_preys, args.num_predators, args.env_x, args.env_y, args.eat_distance, args.confusion)
s.load_prey_genotype(prey)
s.load_predator_genotype(pred)

successes, failures = pygame.init()
print("{0} successes and {1} failures".format(successes, failures))

screen = pygame.display.set_mode((args.env_x, args.env_y))
clock = pygame.time.Clock()

prey_sprite = pygame.Surface((10, 10))
prey_sprite.fill(WHITE)
pygame.draw.circle(prey_sprite, BLUE, (5, 5), 4)

pred_sprite = pygame.Surface((15, 15))
pred_sprite.fill(WHITE)
pygame.draw.circle(pred_sprite, RED, (7, 7), 8)
pygame.draw.line(pred_sprite, BLACK, (7, 7), (7, 0), 2)

FPS = args.fps

count = args.num_preys
init_count = args.num_preys
fit_prey = 0
fit_pred = 0

old = [pos[2] for pos in s.get_predators_pos()]

for timestep in range(args.timesteps):
    screen.fill(WHITE)
    clock.tick(FPS)

    s.step()
    preys_pos = s.get_preys_pos()
    preds_pos = s.get_predators_pos()

    fit_prey += len(preys_pos)
    fit_pred += (init_count - len(preys_pos))

    if len(preys_pos) != count:
        count = len(preys_pos)

    for prey in preys_pos:
        screen.blit(prey_sprite, (int(prey[0]), int(prey[1])))

    for index, pred in enumerate(preds_pos):
        if old[index] - pred[2] > 0:
            sp = pygame.transform.rotate(pred_sprite, -1 * np.degrees(pred[2]))
        else:
            sp = pygame.transform.rotate(pred_sprite, np.degrees(pred[2]))
        screen.blit(sp, (int(pred[0]), int(pred[1])))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()

    pygame.display.update()  # Or pygame.display.flip()

    if len(preys_pos) == 0:
        break

print("FITNESS PREY : ", fit_prey)
print("FITNESS PRED : ", fit_pred)
