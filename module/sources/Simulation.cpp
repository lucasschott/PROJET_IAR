#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <random>
#include "Simulation.hpp"
#include <cmath>
#include <numeric>
#include "utils.hpp"

namespace bp = boost::python;
namespace bn = boost::python::numpy;


Simulation::Simulation(int input_size, int output_size, int num_preys,
		       int num_predators, int env_x, int env_y,
		       double eat_distance, bool confusion)
	: prey_net(Net(input_size * 2, output_size)),
	predator_net(Net(input_size, output_size))
{

    this->num_preys = num_preys;
    this->num_predators = num_predators;
    this->env_x = env_x;
    this->env_y = env_y;
    this->input_size = input_size;
    this->output_size = output_size;
    this->eat_distance = eat_distance;
    this->confusion = confusion;

    this->init_population();
}


void Simulation::init_population()
{
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution_dir(0.0, 2 * M_PI);
	std::uniform_real_distribution<double> distribution_xpos(0.0, env_x);
	std::uniform_real_distribution<double> distribution_ypos(0.0, env_y);

	double x_pos;
	double y_pos;
	double direction;

	for (int i = 0; i < this->num_preys; i++)
	{
	    x_pos = distribution_xpos(generator);
	    y_pos = distribution_ypos(generator);
	    direction = distribution_dir(generator);

	    this->preys.push_back(Individual(PREY, x_pos, y_pos, direction));
	}

	for (int i = 0; i < this->num_predators; i++)
	{
	    x_pos = distribution_xpos(generator);
	    y_pos = distribution_ypos(generator);
	    direction = distribution_dir(generator);

	    this->predators.push_back(Individual(PREDATOR, x_pos,
						 y_pos, direction));
	}
}

void Simulation::clear_population()
{
	this->preys.clear();
	this->predators.clear();
}

void Simulation::step()
{
	std::vector<int> prey_actions;
	std::vector<int> predator_actions;

	this->compute_prey_observations();
	prey_actions = this->forward_prey();

	this->compute_predator_observations();
	predator_actions = this->forward_predator();


	this->apply_prey_actions(prey_actions);
	this->apply_predator_actions(predator_actions);
	this->eat_prey();
	this->clear_population_observations();
}

bp::list Simulation::run(int timesteps)
{
	std::vector<double> current_results;
	double fitness_prey = 0;
	double fitness_pred = 0;
	double density = 0;
	double dispersion = 0;
	int i = 0;

	for (i = 0; i < timesteps && this->preys.size() > 0; i++)
	{
		this->step();
		current_results = this->compute_swarm_density_and_dispersion();
		fitness_prey += this->preys.size();
		fitness_pred += this->num_preys - this->preys.size();

		density += current_results[0];
		dispersion += current_results[1];

		if (this->preys.size() == 0)
		    fitness_pred += this->num_preys * (timesteps - i);
	}

	std::vector<double> results;
	results.push_back(density / i);
	results.push_back(dispersion / i);
	results.push_back(fitness_prey);
	results.push_back(fitness_pred);
	results.push_back(this->preys.size());

	return toPythonList(results);

}

void Simulation::compute_prey_observations()
{
    for (std::vector<Individual>::iterator current = preys.begin() ;
	 current != this->preys.end(); current++)
    {

	    (*current).compute_normal_view_point(this->env_x, this->env_y, 0,
						 0);

	    for (std::vector<Individual>::iterator other = preys.begin() ;
		 other != this->preys.end(); other++)
	    {
		    if (*current == *other)
			    continue;

		    (*current).observe((*other));
	    }

	    for (std::vector<Individual>::iterator other = predators.begin() ;
		 other != this->predators.end(); other++)
	    {
		    (*current).observe((*other));
	    }
    }

}

void Simulation::compute_predator_observations()
{
    for (std::vector<Individual>::iterator current = predators.begin() ;
	 current != this->predators.end(); current++)
    {

	    (*current).compute_normal_view_point(this->env_x, this->env_y, 0,
						 0);

	    for (std::vector<Individual>::iterator other = preys.begin() ;
		 other != this->preys.end(); other++)
	    {
		    (*current).observe((*other));
	    }

	    for (std::vector<Individual>::iterator other = predators.begin() ;
		 other != this->predators.end(); other++)
	    {

		    if (*current == *other)
			    continue;

		    (*current).observe((*other));
	    }

    }
}

std::vector<double> Simulation::compute_swarm_density_and_dispersion()
{
	std::vector<double> results;
	double density = 0;
	double dispersion = 0;

	if (this->preys.size() == 0)
	{
		results.push_back(0);
		results.push_back(0);
		return results;
	}

	for (std::vector<Individual>::iterator current = preys.begin() ;
		current != this->preys.end(); current++)
	{
		density += (*current).get_density();
		dispersion += (*current).get_nearest();
		(*current).clear_density();
		(*current).clear_nearest();
	}

	results.push_back(density / this->preys.size());
	results.push_back(dispersion / this->preys.size());

	return results;
}

void Simulation::reset_population()
{
	this->clear_population();
	this->init_population();
}

bp::list Simulation::get_individuals_pos(std::vector<Individual> vector)
{
	bp::list list;
	std::vector<Individual>::iterator iter;

	for (iter = vector.begin(); iter != vector.end(); ++iter)
	{
		std::vector<double> ind_pos;

		Individual ind = *iter;
		ind_pos.push_back(ind.get_pos_x());
		ind_pos.push_back(ind.get_pos_y());
		ind_pos.push_back(ind.get_direction());

		list.append(toPythonList(ind_pos));
	}

	return list;
}

bp::list Simulation::get_predators_pos()
{
	return this->get_individuals_pos(this->predators);
}

bp::list Simulation::get_preys_pos()
{
	return this->get_individuals_pos(this->preys);
}

void Simulation::load_predator_genotype(bp::list genotype)
{
	std::vector<float> imported = to_std_vector<float>(genotype);
	this->predator_net.load_from_vector(imported);
}

void Simulation::load_prey_genotype(bp::list genotype)
{
	std::vector<float> imported = to_std_vector<float>(genotype);
	this->prey_net.load_from_vector(imported);
}

std::vector<int> Simulation::forward_prey()
{
	torch::Tensor batch =
		torch::zeros({int(this->preys.size()), 2 * this->input_size});
	int index = 0;
	std::vector<Individual>::iterator iter;
	std::vector<int> converted;

	for (iter = this->preys.begin(); iter != this->preys.end(); iter++)
	{
		batch[index] =
			torch::tensor((*iter).get_observations(), at::kFloat);
		index = index + 1;
	}

	torch::Tensor actions = at::argmax(this->prey_net.forward(batch), 1);


	for (int i = 0; i < actions.size(0); i++)
		converted.push_back(actions[i].item().toInt());

	return converted;
}

std::vector<int> Simulation::forward_predator()
{
	torch::Tensor batch =
		torch::zeros({int(this->predators.size()), this->input_size});
	int index = 0;
	std::vector<Individual>::iterator iter;
	std::vector<int> converted;

	for (iter = this->predators.begin(); iter != this->predators.end();
	     iter++)
	{
		batch[index] =
			torch::tensor((*iter).get_observations(), at::kFloat);
		index = index + 1;
	}

	torch::Tensor actions =
		at::argmax(this->predator_net.forward(batch), 1);

	for (int i = 0; i < actions.size(0); i++)
		converted.push_back(actions[i].item().toInt());

	return converted;
}

void Simulation::clear_population_observations()
{
	std::vector<Individual>::iterator iter;
	for (iter = this->predators.begin(); iter != this->predators.end();
	     iter++)
	{
		(*iter).clear_observations();
	}

	for (iter = this->preys.begin(); iter != this->preys.end();
	     iter++)
	{
		(*iter).clear_observations();
	}
}

void Simulation::apply_prey_actions(std::vector<int> &actions)
{
	int index = 0;

	std::vector<Individual>::iterator iter;
	for (iter = this->preys.begin(); iter != this->preys.end();
	     iter++)
	{
		(*iter).apply_action(actions[index]);
		index = index + 1;
	}
}

void Simulation::apply_predator_actions(std::vector<int> &actions)
{
	int index = 0;

	std::vector<Individual>::iterator iter;

	for (iter = this->predators.begin(); iter != this->predators.end();
	     iter++)
	{
		(*iter).apply_action(actions[index]);
		index = index + 1;
	}
}

bool Simulation::get_eat_flag(std::vector<int> observation)
{
	double sum = std::accumulate(observation.begin(), observation.end(),
				     0.0);
	double prob = 1 / sum;
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0, 1);

	return  distribution(generator) <= prob;
}

void Simulation::eat_prey()
{
	std::vector<Individual>::iterator iter_pred;
	std::vector<Individual>::iterator iter_prey;
	int last_meal;
	bool eat;

	for (iter_pred = this->predators.begin();
	     iter_pred != this->predators.end(); iter_pred++)
	{
		last_meal = (*iter_pred).get_last_meal();
		(*iter_pred).set_last_meal(last_meal + 1);

		if (last_meal < 10)
			continue;

		for (iter_prey = this->preys.begin(); iter_prey != this->preys.end();)
		{


			if ((*iter_pred).get_last_meal() < 10)
				break;

			if ((*iter_pred).get_distance_to(*iter_prey) <
			    this->eat_distance)
			{
				if (this->confusion == false)
					eat = true;
				else
					eat = get_eat_flag((*iter_pred).get_observations());

				if (eat)
				{
					iter_prey = this->preys.erase(iter_prey);
					(*iter_pred).set_last_meal(0);
				}

				else
					iter_prey++;
			}

			else

				iter_prey++;
		}

	}
}
