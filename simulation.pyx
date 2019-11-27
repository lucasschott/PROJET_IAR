import time
import math
import numpy as np
import collections
import matplotlib.pyplot as plt
import copy


import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cuda = False

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    cuda = True
    print("GPU MODE ENABLED")
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    print("CPU MODE ENABLED")


class Net(nn.Module):

    def __init__(self,input,output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input,12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 12)
        self.fc4 = nn.Linear(12, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc3(x)
        return x


def dict_to_list(my_dict):
    my_list = []
    my_list.append(my_dict["fc1.weight"].tolist())
    my_list.append(my_dict["fc1.bias"].tolist())
    my_list.append(my_dict["fc2.weight"].tolist())
    my_list.append(my_dict["fc2.bias"].tolist())
    my_list.append(my_dict["fc3.weight"].tolist())
    my_list.append(my_dict["fc3.bias"].tolist())
    my_list.append(my_dict["fc4.weight"].tolist())
    my_list.append(my_dict["fc4.bias"].tolist())
    w1,b1,w2,b2,w3,b3,w4,b4 = my_list
    w1 = np.array(w1).flatten().tolist()
    w2 = np.array(w2).flatten().tolist()
    w3 = np.array(w3).flatten().tolist()
    w4 = np.array(w4).flatten().tolist()
    return w1 + b1 + w2 +b2 + w3 + b3 + w4 + b4

def list_to_dict(input,my_list):
    my_list = np.array(my_list)
    my_dict = collections.OrderedDict()
    a = 0
    b = input*12
    my_dict["fc1.weight"] = torch.Tensor(my_list[a:b].reshape(12,input))
    a = b
    b += 12
    my_dict["fc1.bias"] = torch.Tensor(my_list[a:b])
    a = b
    b += 12*12
    my_dict["fc2.weight"] = torch.Tensor(my_list[a:b].reshape(12,12))
    a = b
    b += 12
    my_dict["fc2.bias"] = torch.Tensor(my_list[a:b])
    a = b
    b += 12*12
    my_dict["fc3.weight"] = torch.Tensor(my_list[a:b].reshape(12,12))
    a = b
    b += 12
    my_dict["fc3.bias"] = torch.Tensor(my_list[a:b])
    a = b
    b += 12*4
    my_dict["fc4.weight"] = torch.Tensor(my_list[a:b].reshape(4,12))
    a = b
    b += 4
    my_dict["fc4.bias"] = torch.Tensor(my_list[a:b])
    return my_dict




def dist(p1,p2):
    return math.sqrt( (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 )


class ENV():

    def __init__(self,confusion):
        #agents
        self.num_preys = 50
        self.swarm_density = 0
        self.swarm_dispersion = 0
        self.sensor_angle = math.radians(15)

        #pred
        self.confusion = confusion
        self.pred_size = 15
        self.last_meal = 10
        self.velocity_pred = 9
        self.rotation_pred = math.radians(6)
        self.num_sensor_pred = 12
        self.sensor_distance_pred = 200
        self.digest_time_pred = 10
        self.input_pred = list(np.zeros(12))

        self.position_pred = np.random.randint(low=0,high=512,size=(2))
        self.direction_pred = np.random.random_sample()*2*math.pi

        #prey
        self.prey_size = 9
        self.velocity_prey = 3
        self.rotation_prey = 8
        self.num_sensor_prey = 24
        self.sensor_distance_prey = 100
        self.input_preys = [list(np.zeros(12)) for i in range(self.num_preys)]

        self.position_preys = np.random.randint(low=0,high=512,size=(50,2))
        self.direction_preys = np.random.random_sample((50))*2*math.pi

        #env
        self.x_size= 512
        self.y_size= 512
        self.map = np.zeros((self.x_size,self.y_size),dtype=np.intc)



    def ray_casting(self,target,position,distance,self_size,step_size,ray_direction):
        real_pos = copy.deepcopy(position)
        first_step = (self_size*np.cos(ray_direction),self_size*np.sin(ray_direction))
        real_pos[0] += first_step[0]
        real_pos[0] %= 512
        real_pos[1] += first_step[1]
        real_pos[1] %= 512
        step = (step_size*np.cos(ray_direction),step_size*np.sin(ray_direction))
        for i in range(distance//step_size):
            int_pos = int(math.floor(real_pos[0])),int(math.floor(real_pos[1]))
            if self.map[int_pos[0],int_pos[1]]==target:
                return 1
            real_pos[0] += step[0]
            real_pos[0] %= 512
            real_pos[1] += step[1]
            real_pos[1] %= 512
        return 0

    """
    def pred_observation(self):
        observation = np.zeros(12)
        for sensor in range(12): #nb sensors
            ray_direction = (self.direction_pred + math.pi/2 - sensor*math.radians(15)) % (2*math.pi)
            for ray in range(3): #nb rays
                ray_direction += math.radians(5)
                ray_direction %= 2*math.pi
                observation[sensor] = self.ray_casting(1,self.position_pred,
                        self.sensor_distance_pred,self.pred_size,
                        self.prey_size,ray_direction)
                if observation[sensor] == 1:
                    break
        return list(observation)
    

    def prey_observation(self,idx):
        observation = np.zeros(24)
        for sensor in range(12): #nb sensors
            ray_direction = (self.direction_preys[idx] + math.pi/2 - sensor*math.radians(15)) % (2*math.pi)
            for ray in range(3): #nb rays
                ray_direction += math.radians(5)
                ray_direction %= 2*math.pi
                observation[sensor] = self.ray_casting(1,self.position_preys[idx],
                        self.sensor_distance_prey,self.prey_size,
                        self.prey_size,ray_direction)
                if observation[sensor] == 1:
                    break
        for sensor in range(12,24): #nb sensors
            ray_direction = (self.direction_preys[idx] + math.pi/2 - (sensor-12)*math.radians(15)) % (2*math.pi)
            for ray in range(1): #nb rays
                ray_direction += math.radians(7)
                ray_direction %= 2*math.pi
                observation[sensor] = self.ray_casting(1,self.position_preys[idx],
                        self.sensor_distance_prey,self.prey_size,
                        self.pred_size,ray_direction)
                if observation[sensor] == 1:
                    break
        return list(observation)
    """

    def pred_observation(self):
        observation = list(np.zeros(12))
        angles = []
        for position_prey in self.position_preys:
            a1 = 1
            a2 = 0
            b1 = self.position_pred[0] - position_prey[0]
            b2 = self.position_pred[1] - position_prey[1]
            if dist(position_prey,self.position_pred) < self.sensor_distance_pred:
                angle = ( self.direction_pred + math.pi/2 - np.arccos((a1*b1+a2*b2)/(1*dist(self.position_pred,position_prey))) ) % (2*math.pi)
                i = int(angle // self.sensor_angle)
                if i >=0 and i<12:
                    observation[i] = 1
        return observation

    def prey_observation(self,idx):
        observation_prey = list(np.zeros(12))
        for position_prey in self.position_preys:
            if (self.position_preys[idx] == position_prey).all():
                continue
            a1 = 1
            a2 = 0
            b1 = self.position_preys[idx,0] - position_prey[0]
            b2 = self.position_preys[idx,1] - position_prey[1]
            if dist(position_prey,self.position_preys[idx]) < self.sensor_distance_prey:
                angle = ( self.direction_preys[idx] + math.pi/2 - np.arccos((a1*b1+a2*b2)/(1*dist(self.position_preys[idx],position_prey))) ) % (2*math.pi)
                i = int(angle // self.sensor_angle)
                if i >=0 and i<12:
                    observation_prey[i] = 1
        observation_pred = list(np.zeros(12))
        a1 = 1
        a2 = 0
        b1 = self.position_preys[idx,0] - self.position_pred[0]
        b2 = self.position_preys[idx,1] - self.position_pred[1]
        if dist(self.position_pred,self.position_preys[idx]) < self.sensor_distance_prey:
            angle = ( self.direction_preys[idx] + math.pi/2 - np.arccos((a1*b1+a2*b2)/(1*dist(self.position_preys[idx],self.position_pred))) ) % (2*math.pi)
            i = int(angle // self.sensor_angle)
            if i >=0 and i<12:
                observation_pred[i] = 1

        return observation_prey + observation_pred
    
    def preys_observations(self):
        input_preys = []
        for i in range(self.num_preys):
            input_preys.append(self.prey_observation(i))
        return input_preys


    def compute_swarm_density(self):
        num_prey_30m = []
        for i,position_prey_1 in enumerate(self.position_preys):
            num_prey_30m.append(0)
            for position_prey_2 in self.position_preys:
                if dist(position_prey_1,position_prey_2) < 30:
                    num_prey_30m[i]+=1
        return np.mean(num_prey_30m)


    def compute_swarm_dispersion(self):
        dist_nearest_prey = []
        for i,position_prey_1 in enumerate(self.position_preys):
            dist_nearest_prey.append(1000)
            for position_prey_2 in self.position_preys:
                if dist(position_prey_1,position_prey_2) < dist_nearest_prey[i]:
                    dist_nearest_prey[i] = dist(position_prey_1,position_prey_2)
        return np.mean(dist_nearest_prey)


    def step(self, action_pred, actions_preys):

        if action_pred == 3:
            self.direction_pred += self.rotation_pred
        elif action_pred == 4:
            self.direction_pred -= self.rotation_pred
        self.direction_pred %= 2*math.pi
        if action_pred >= 2:
            self.position_pred[0] += math.cos(self.direction_pred)*self.velocity_pred
            self.position_pred[1] += math.sin(self.direction_pred)*self.velocity_pred
        self.position_pred[0] %= self.x_size
        self.position_pred[1] %= self.y_size

        for i,action_prey in enumerate(actions_preys):
            if action_pred == 3:
                self.direction_preys[i] += self.rotation_prey
            elif action_pred == 4:
                self.direction_preys[i] -= self.rotation_prey
            self.direction_preys[i] %= 2*math.pi
            if action_prey >= 2:
                self.position_preys[i,0] += math.cos(self.direction_preys[i])*self.velocity_prey
                self.position_preys[i,1] += math.sin(self.direction_preys[i])*self.velocity_prey
            self.position_preys[i,0] %= self.x_size
            self.position_preys[i,1] %= self.y_size

        self.update_map()
        
        self.nput_pred = self.pred_observation()

        self.eat_prey()

        self.input_preys = self.preys_observations()

        return self.input_pred, self.input_preys, self.num_preys


    def update_map(self):
        self.map = np.zeros((self.x_size,self.y_size),dtype=np.intc)
        prey_size = self.prey_size//2
        pred_size = self.pred_size//2
        for x,y in self.position_preys:
            for xi in range(x-prey_size, x+prey_size):
                for yi in range(y-prey_size, y+prey_size):
                    self.map[xi%512,yi%512] = 1

        x,y = self.position_pred
        for xi in range(x-pred_size, x+pred_size):
            for yi in range(y-pred_size, y+pred_size):
                self.map[xi%512,yi%512] = 2


    def eat_prey(self):
        self.last_meal +=1
        for i,position_prey in enumerate(self.position_preys):
            if dist(position_prey,self.position_pred)<self.pred_size+self.prey_size and self.last_meal>10:
                if not self.confusion:
                    eat = True
                else:
                    prob = 1/np.sum(self.input_pred)
                    rand  = np.random.random()
                    eat = rand <= prob
                if eat:
                    print("eat {}".format(i))
                    self.position_preys = np.delete(self.position_preys,[i],0)
                    self.direction_preys = np.delete(self.direction_preys,[i],0)
                    self.num_preys -= 1
                    self.last_meal = 0
                    break

    
    def reset(self):
        self.num_preys = 50
        self.position_pred = np.random.randint(low=0,high=512,size=(2))
        self.direction_preys = np.random.random_sample((50))*2*math.pi
        self.position_preys = np.random.randint(low=0,high=512,size=(50,2))
        self.direction_preys = np.random.random_sample((50,2))*2*math.pi

    




def simulation(pred_indiv,prey_indiv,confusion,log,render=True):

    t1 = time.time()

    pred_nn = Net(input=12,output=4)
    prey_nn = Net(input=24,output=4)

    print("NEW SIMULATION")

    pred_indiv = list_to_dict(12,pred_indiv)
    prey_indiv = list_to_dict(24,prey_indiv)

    pred_nn.load_state_dict(pred_indiv,strict=False)
    prey_nn.load_state_dict(prey_indiv,strict=False)

    fitness_pred = 0
    fitness_prey = 0
    
    swarm_densitys = []
    swarm_dispersions = []

    env = ENV(confusion)

    input_pred = torch.Tensor(env.pred_observation())
    input_preys = torch.Tensor(env.preys_observations())

    if render:
        plt.imshow(env.map)
        plt.pause(0.01)

    for step in range(2000):
        
        with torch.set_grad_enabled(False):
            pred_output = pred_nn.forward(input_pred)
            preys_outputs = prey_nn.forward(input_preys)
        pred_action = np.argmax(pred_output)
        preys_actions = np.argmax(preys_outputs,axis=1)

        input_pred, input_preys, survivorship  = env.step(pred_action, preys_actions)

        if survivorship == 0:
            break

        if log:
            swarm_density = env.compute_swarm_density()
            swarm_dispersion = env.compute_swarm_dispersion()
            swarm_densitys.append(swarm_density)
            swarm_dispersions.append(swarm_dispersion)

        input_pred = torch.Tensor(input_pred)
        input_preys = torch.Tensor(input_preys)

        fitness_pred += 200 - survivorship
        fitness_prey += survivorship
        
        if render:
            plt.imshow(env.map)
            plt.pause(0.01)
        
    if render:
        plt.show()

    t2 = time.time()
    print("time step = {}".format(t2-t1))
    

    if log:
        return fitness_pred, fitness_prey, survivorship, np.mean(swarm_density), np.mean(swarm_dispersion)
    else:
        return fitness_pred, fitness_prey



