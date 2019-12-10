#include <iostream>
#include <vector>

#define PREY "prey"
#define PREDATOR "predator"
#define NB_BINS 12

#define UP 0
#define DOWN 1
#define LEFT 2
#define RIGHT 3

class Individual
{

public:
	Individual(std::string type, double pos_x, double pos_y, double direction,
		   int env_x, int env_y);

	double get_pos_x();
	double get_pos_y();
	double get_direction();
	double get_density();
	double get_nearest();


	void set_pos_x(int new_pos_x);
	void set_pos_y(int new_pos_y);
	void set_direction(double new_direction);
	void clear_observations();
	std::vector<int> get_observations();

	Individual get_repositioned_individual(Individual &other);
	void observe(Individual &other);
	void apply_action(int action);
	void compute_flags();
	double get_distance_to(Individual &other);
	double get_angle_to(Individual &other);
	void compute_normal_view_point(double env_x_max, double env_y_max,
					     double env_x_min,
					     double env_y_min);
	void clear_density();
	void clear_nearest();
	int get_last_meal();
	void set_last_meal(int last_meal);

	inline bool operator==(const Individual& other)
	{
		return this->id == other.id;
	}

private:
	std::string type;

	double view_distance;
	double direction;
	double pos_x;
	double pos_y;
	double rotation;
	double velocity;
	double nearest;
	double density;

	double view_point[2];
	double view_vector[2];

	bool right;
	bool left;
	bool up;
	bool down;

	int id;
	int last_meal;
	int env_x;
	int env_y;

	bool is_alive;
	std::vector<int> observations;
};
