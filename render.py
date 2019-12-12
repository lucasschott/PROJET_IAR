import numpy as np
import time
import cma
import argparse
from functools import partial
import matplotlib.pyplot as plt
import pygame

import multiprocessing as mp
from module.build import pylib

net_input = 12
net_output = 4

LAYER = 12
PRED_NETWORK_SIZE = net_input*LAYER+LAYER + LAYER*net_output+net_output
PREY_NETWORK_SIZE = (net_input*2)*LAYER+LAYER + LAYER*net_output+net_output

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

screen = pygame.display.set_mode((args.env_x, args.env_y + 80))
clock = pygame.time.Clock()

pred_vision = pygame.Surface((400, 400))
pred_vision.fill(WHITE)
pred_vision.set_colorkey(WHITE)
pygame.draw.circle(pred_vision, BLACK, (200, 200), 200, 1)

prey_sprite = pygame.Surface((10, 10))
prey_sprite.fill(WHITE)
prey_sprite.set_colorkey(WHITE)
pygame.draw.circle(prey_sprite, BLUE, (5, 5), 4)

pred_sprite = pygame.Surface((400, 400))
pred_sprite.fill(WHITE)
pred_sprite.set_colorkey(WHITE)
pygame.draw.circle(pred_sprite, BLACK, (200, 200), 200, 1)
pygame.draw.rect(pred_sprite, WHITE, (0, 0, 400, 200))
pygame.draw.circle(pred_sprite, RED, (200, 200), 8)

obs_bg = pygame.Surface((args.env_x, 80))
obs_bg.fill(BLACK)

obs_activated = pygame.Surface((20, 20))
obs_activated.fill(BLACK)

pygame.draw.circle(obs_activated, WHITE, (10, 10), 8)
pygame.draw.circle(obs_activated, RED, (10, 10), 6)

obs_idle = pygame.Surface((20, 20))
obs_idle.fill(BLACK)
pygame.draw.circle(obs_idle, WHITE, (10, 10), 8)

FPS = args.fps

count = args.num_preys
init_count = args.num_preys
fit_prey = 0
fit_pred = 0

old = [pos[2] for pos in s.get_predators_pos()]

pred_obs = None

def pred_hook(observations):
    global pred_obs
    pred_obs = observations[0]

def prey_hook(observations):
    pass

for timestep in range(args.timesteps):
    screen.fill(WHITE)
    clock.tick(FPS)

    s.step_hook(pred_hook, prey_hook)
    preys_pos = s.get_preys_pos()
    preds_pos = s.get_predators_pos()

    fit_prey += len(preys_pos)
    fit_pred += (init_count - len(preys_pos))

    if len(preys_pos) != count:
        count = len(preys_pos)

    for prey in preys_pos:
        screen.blit(prey_sprite, (int(prey[0]) - prey_sprite.get_width() / 2, int(prey[1]) - prey_sprite.get_height() / 2))

    for index, pred in enumerate(preds_pos):
        sp = pygame.transform.rotate(pred_sprite, -1 *  np.degrees(pred[2]) + 90)
        screen.blit(sp, (int(pred[0] - sp.get_width() / 2), int(pred[1]) - sp.get_height() / 2))
        end_x = pred[0] + np.cos(pred[2]) * 200
        end_y = pred[1] + np.sin(pred[2]) * 200

        pygame.draw.line(screen, BLUE, (pred[0], pred[1]), (end_x, end_y), 1)

        end_x = pred[0] + np.cos(pred[2] - np.pi / 2) * 200
        end_y = pred[1] + np.sin(pred[2] - np.pi / 2) * 200

        pygame.draw.line(screen, RED, (pred[0], pred[1]), (end_x, end_y), 1)

        end_x = pred[0] + np.cos(pred[2] + np.pi / 2) * 200
        end_y = pred[1] + np.sin(pred[2] + np.pi / 2) * 200

        pygame.draw.line(screen, RED, (pred[0], pred[1]), (end_x, end_y), 1)

    screen.blit(obs_bg, (0, args.env_y))

    for index, obs in enumerate(pred_obs):

        if obs == 1:
            sprite = obs_activated
        else:
            sprite = obs_idle

        screen.blit(sprite, ((args.env_x / args.input) * index + sprite.get_width() / 2, args.env_y + 40 - sprite.get_height() / 2))



    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()

        if event.type ==  pygame.KEYDOWN:

            if event.key == pygame.K_DOWN:
                FPS = max(1, FPS - 10)

            if event.key == pygame.K_UP:
                FPS = FPS + 10

    pygame.display.update()  # Or pygame.display.flip()

    if len(preys_pos) == 0:
        break

print("FITNESS PREY : ", fit_prey)
print("FITNESS PRED : ", fit_pred)
