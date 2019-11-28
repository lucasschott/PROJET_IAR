#include <stdlib.h>
#include <iostream>
#include <time.h>
#include "boost/python/numpy.hpp"
#include "Simulation.hpp"
#include "utils.hpp"

namespace bp = boost::python;
namespace bn = boost::python::numpy;

Simulation::Simulation(std::string mode, int num_preys, int num_predators,
	       int env_x, int env_y)
{
    // std::cout << " Mode : " << mode << std::endl;
    // std::cout << " Preys : " << num_preys << std::endl;
    // std::cout << " Predator : " << num_predators << std::endl;
    // std::cout << "Env x : " << env_x << std::endl;
    // std::cout << "Env y : " << env_y << std::endl;

    this->mode = mode;
    this->num_preys = num_preys;
    this->num_predators = num_predators;
    this->env_x = env_x;
    this->env_y = env_y;

    srand(time(NULL));
    this->init_population();
}

void Simulation::init_population()
{
    for (unsigned int i = 0; i < this->num_preys; i++)
    {
	    int x_pos = rand() % this->env_x;
	    int y_pos = rand() % this->env_y;

	    // std::cout << "Prey : [" << i << "] x = " << x_pos << " y = " << y_pos << std::endl;

	    this->preys.push_back(Individual(PREY, x_pos, y_pos));
    }

    for (unsigned int i = 0; i < this->num_predators; i++)
    {
	    int x_pos = rand() % this->env_x;
	    int y_pos = rand() % this->env_y;

	    // std::cout << "Predator : [" << i << "] x = " << x_pos << " y = " << y_pos << std::endl;

	    this->predators.push_back(Individual(PREDATOR, x_pos,
						 y_pos));
    }
}

void Simulation::clear_population()
{
	this->preys.clear();
	this->predators.clear();
}

void Simulation::reset_population()
{
	this->clear_population();
	this->init_population();
}

void Simulation::set_mode(std::string mode)
{
	this->mode = mode;
}

std::string Simulation::get_mode() const
{
	return this->mode;
}

bp::list Simulation::get_individuals_pos(std::vector<Individual> vector)
{
	bp::list list;
	std::vector<Individual>::iterator iter;

	for (iter = vector.begin(); iter != vector.end(); ++iter)
	{
		std::vector<int> ind_pos;

		Individual ind = *iter;
		ind_pos.push_back(ind.get_pos_x());
		ind_pos.push_back(ind.get_pos_y());

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
