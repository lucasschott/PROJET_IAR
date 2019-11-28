#include <iostream>

#define PREY "prey"
#define PREDATOR "predator"

class Individual
{

public:
	Individual(std::string type, int pos_x, int pos_y);

	int get_pos_x();
	int get_pos_y();

	void set_pos_x(int new_pos_x);
	void set_pos_y(int new_pos_y);


private:
	std::string type;
	int pos_x;
	int pos_y;
	bool is_alive;
};
