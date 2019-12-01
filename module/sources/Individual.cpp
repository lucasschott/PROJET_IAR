#include <cmath>
#include <algorithm>
#include <limits>
#include "Individual.hpp"

unsigned int generate_new_id()
{
	static unsigned int current = 0;
	unsigned int id = current;

	current++;
	return id;

}

double get_orientation(double *vec1, double *vec2)
{
	double sign = -vec1[0] * vec2[1] + vec1[1] * vec2[0];

	if (sign >= 0)
		return 1;

	return -1;
}

double get_norm(double *vec, int dim)
{
	double sum = 0;

	for (int i = 0; i < dim; i++)
		sum += pow(vec[i], 2);

	return sqrt(sum);
}

Individual::Individual(std::string type, double pos_x, double pos_y,
		       double direction)
{
	this->is_alive = true;
	this->type = type;
	this->pos_x = pos_x;
	this->pos_y = pos_y;
	this->id = generate_new_id();
	this->direction = direction;
	this->density = 0;
	this->nearest = std::numeric_limits<double>::infinity();
	this->last_meal = 10;

	if (type == PREDATOR)
	{
		this->view_distance = 200;
		this->rotation = (6.0 * M_PI) / 180.0;
		this->velocity = 9;
		this->observations = std::vector<int>(NB_BINS, 0);
	}

	else
	{
		this->view_distance = 100;
		this->rotation = (8.0 * M_PI) / 180.0;
		this->velocity = 3;
		this->observations = std::vector<int>(2 * NB_BINS, 0);
	}

}

double Individual::get_pos_x()
{
	return this->pos_x;
}

double Individual::get_pos_y()
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

void Individual::clear_observations()
{
	std::fill(this->observations.begin(), this->observations.end(), 0);
}

void Individual::clear_density()
{
	this->density = 0;
}

void Individual::clear_nearest()
{
	this->nearest = std::numeric_limits<double>::infinity();
}

void Individual::compute_normal_view_point(double env_x_max, double env_y_max,
					     double env_x_min, double env_y_min)
{
	double v_x = this->pos_x + this->view_distance * cos(this->direction);
	double v_y = this->pos_y + this->view_distance * sin(this->direction);

	this->view_point[0] = std::min(env_x_max, std::max(env_x_min, v_x));
	this->view_point[1] = std::min(env_y_max, std::max(env_y_min, v_y));

	this->view_vector[0] = this->view_point[0] - this->pos_x;
	this->view_vector[1] = this->view_point[1] - this->pos_y;
}

double Individual::get_distance_to(Individual &other)
{
	double diff_x = this->pos_x - other.pos_x;
	double diff_y = this->pos_y - other.pos_y;

	return sqrt(pow(diff_x, 2) + pow(diff_y, 2));
}

double Individual::get_angle_to(Individual &other)
{
	double view_to_other[2];
	double me_to_other[2];

	double a;
	double c;
	double b;

	double angle;

	view_to_other[0] = other.get_pos_x() - this->view_point[0];
	view_to_other[1] = other.get_pos_y() - this->view_point[1];
	me_to_other[0] = other.get_pos_x() - this->pos_x;
	me_to_other[1] = other.get_pos_y() - this->pos_y;

	b = get_norm(this->view_vector, 2);
	c = get_norm(me_to_other, 2);
	a = get_norm(view_to_other, 2);

	angle = (pow(b, 2) + pow(c, 2) - pow(a, 2)) / (2 * c * b);
	angle = acos(angle);

	return angle * get_orientation(this->view_vector, me_to_other);

}

void Individual::observe(Individual &other)
{
	double distance = get_distance_to(other);
	double angle;
	int bin;

	if (distance < this->nearest)
		this->nearest = distance;

	if (distance > this->view_distance)
		return;

	angle = this->get_angle_to(other);

	// Translate to [0 PI/2] and switch to degrees
	angle = (angle + M_PI / 2) * 180 / M_PI;
	bin = int(angle / (180 / NB_BINS));

	if (bin >= 0 && bin < NB_BINS)
	{
		if (other.type == PREDATOR)
			bin = bin + NB_BINS;

		else if (distance < 30)
			this->density += 1;

		this->observations[bin] = 1;
	}
}

std::vector<int> Individual::get_observations()
{
	return this->observations;
}

double Individual::get_density()
{
	return this->density;
}

double Individual::get_nearest()
{
	return this->nearest;
}

int Individual::get_last_meal()
{
	return this->last_meal;
}

void Individual::set_last_meal(int last_meal)
{
	this->last_meal = last_meal;
}

void Individual::apply_action(int action)
{
	if (action >= 1)
	{
		this->pos_x += cos(this->direction) * this->velocity;
		this->pos_y += sin(this->direction) * this->velocity;
		this->pos_x = std::min(500.0,std::max(0.0, this->pos_x));
		this->pos_y = std::min(500.0,std::max(0.0, this->pos_y));
	}

	if (action == 2)
	{
		this->direction += this->rotation;
		this->direction = fmod(this->direction, 2 * M_PI);
	}

	if (action == 3)
	{
		this->direction -= this->rotation;
		this->direction = fmod(this->direction, 2 * M_PI);
	}
}
