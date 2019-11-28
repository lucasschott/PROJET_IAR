#include "Individual.hpp"

Individual::Individual(std::string type, int pos_x, int pos_y)
{
	this->is_alive = true;
	this->type = type;
	this->pos_x = pos_x;
	this->pos_y = pos_y;
}

int Individual::get_pos_x()
{
	return this->pos_x;
}

int Individual::get_pos_y()
{
	return this->pos_y;
}

void Individual::set_pos_x(int new_pos_x)
{
	this->pos_x = new_pos_x;
}

void Individual::set_pos_y(int new_pos_y)
{
	this->pos_y = new_pos_y;
}

