from build import pylib

mode = "ray"
num_preys = 10
num_predators = 2
env_x = 500
env_y = 500

s = pylib.Simulation(mode, num_preys, num_predators, env_x, env_y)

print("PREY POSITIONS : ", s.get_preys_pos())
print("PREDATOR POSITIONS : ", s.get_predators_pos())

