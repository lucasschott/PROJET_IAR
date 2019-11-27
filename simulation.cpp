#include <cstdlib>
#include <torch/torch.h>

#define PI 3.14159265

using namespace std;



/********** functions **********/
/*******************************/


float rad(float x){
    return x*PI/180
}




/********** Neural Network **********/
/************************************/


class Net : torch::nn::Module{
    auto fc1, fc2, fc3, fc4;

   Net(input,output);
   torch::Tensor forward(x);
};


Net::Net(auto input,auto output){
    fc1 = torch::nn::Linear(input,12);
    fc2 = torch::nn::Linear(12, 12);
    fc3 = torch::nn::Linear(12, 12);
    fc4 = torch::nn::Linear(12, output);
}


torch::Tensor Net::forward(torch::Tensor x){
    x = F.relu(self.fc1(x));
    x = F.relu(self.fc2(x));
    x = F.relu(self.fc3(x));
    x = self.fc3(x);
    return x;
}




/********** Environment **********/
/*********************************/


class Env{
    int num_preys,sensor_angle;
    float swarm_density,swarm_dispersion;

    bool confusion;
    int pred_size,last_meal,velocity_pred,num_sensor_pred,
        sensor_distance_pred,digest_time_pred;
    float rotation_pred,direction_pred;
    int input_pred[12];
    int position_pred[2];

    int prey_size,velocity_prey,num_sensor_prey,sensor_distance_prey
    float rotation_prey,direction_preys;
    int ** input_preys;
    auto position_preys;

    int x_size,y_size;
    int ** env_map;

    Env(bool confusion);
    int ray_casting(int target, auto position, int distance, int self_size,
            int step_size, auto ray_direction);
    int * pred_observation();
    int * prey_observation();
    int ** preys_observations();
    float compute_swarm_density();
    float compute_swarm_dispersion();
    auto step(auto action_pred, auto action_preys);
    void reset_env_map();
    void eat_prey();
    void reset();
};


Env::Env(bool confusion){
    //agents
    self.num_preys = 50
    self.swarm_density = 0
    self.swarm_dispersion = 0
    self.sensor_angle = rad(15)

    //pred
    self.confusion = confusion
    self.pred_size = 11
    self.last_meal = 10
    self.velocity_pred = 3
    self.rotation_pred = rad(6)
    self.num_sensor_pred = 12
    self.sensor_distance_pred = 200
    self.digest_time_pred = 10
    self.input_pred = {0,0,0,0,0,0,0,0,0,0,0,0}
    for (int i=0, i<12, i++){
        self.input_pred[i] = 0
    }

    self.position_pred[0] = rand() % 512
    self.position_pred[1] = rand() % 512
    self.direction_pred = (rand()/(RAND_MAX+1.0f))*2*PI

    //prey
    self.prey_size = 7
    self.velocity_prey = 1
    self.rotation_prey = 8
    self.num_sensor_prey = 24
    self.sensor_distance_prey = 100
    for (int i=0, i<self.num_preys, i++){
        for (int j=0, i<12, i++){
            self.input_preys[i][j] = 0
        }
    }
    self.input_preys = [list(np.zeros(12)) for i in range(self.num_preys)]

    self.position_preys = np.random.randint(low=0,high=512,size=(50,2))
    self.direction_preys = np.random.randint(low=0,high=2*math.pi,size=(50,1))

    //env
    self.x_size= 512
    self.y_size= 512
    self.env_map = np.zeros((self.x_size,self.y_size),dtype=np.intc)
    self.counter = 0
}

