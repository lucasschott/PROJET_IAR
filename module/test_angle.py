# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 13:29:19 2019

@author: Hector
"""


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

env_size = 500
view_distance = 200

fig, ax=plt.subplots()

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

def get_new_point(point1, point2):

    flags_1 = [point1[1] > env_size - view_distance,
               point1[1] < view_distance,
               point1[0] < view_distance,
               point1[0] > env_size - view_distance]

    flags_2 = [point2[1] > env_size - view_distance,
               point2[1] < view_distance,
               point2[0] < view_distance,
               point2[0] > env_size - view_distance]

    new_point = [point2[0], point2[1]]

    if flags_1[LEFT] and new_point[0] > env_size - view_distance:
        new_point[0] -= env_size

    elif flags_1[RIGHT] and new_point[0] < view_distance:
        new_point[0] += env_size

    if flags_1[DOWN] and new_point[1] > env_size - view_distance:
        new_point[1] -= env_size

    elif flags_1[UP] and new_point[1] < view_distance:
        new_point[1] += env_size

    return np.array(new_point)

point_1 = np.array([450, 450])
point_2_ori = np.random.rand(2)
point_2 = get_new_point(point_1, point_2_ori)

direction_1 = np.random.rand() * 2 * np.pi
direction_2 = 0

def compute_max_view_point(point, view_distance, direction):
    v_x = (point_1[0] + view_distance * np.cos(direction_1))
    v_y = (point_1[1] + view_distance * np.sin(direction_1))

    v_x = min(env_size, max(0, v_x))
    v_y = min(env_size, max(0, v_y))

    return v_x, v_y

def plot_line(point1, point2):
    x = [point1[0], point2[0]]
    y = [point1[1], point2[1]]
    plt.plot(x, y, marker="o")

def points_to_vec(p1, p2):
    return p2 - p1

def get_angle_vec_3(o_to_view, view_to_p, o_to_p):

    """
    b = o_to_view
    a = view_to_p
    c = o_to_p
    """

    b = np.linalg.norm(o_to_view)
    a = np.linalg.norm(view_to_p)
    c = np.linalg.norm(o_to_p)

    cos_angle =  (b ** 2 + c ** 2 - a ** 2) / (2 * c * b)

    orientation = -o_to_view[0] * o_to_p[1] + o_to_view[1] * o_to_p[0]

    print(np.sign(orientation))

    return np.arccos(cos_angle)

max_view_point = compute_max_view_point(point_1, view_distance, direction_1)

ax.plot(point_1[0], point_1[1], "r+")
ax.annotate("Point_1", (point_1[0], point_1[1]))

ax.plot(point_2_ori[0], point_2_ori[1], "r+")
ax.annotate("Point_2_ori", (point_2_ori[0], point_2_ori[1]))

ax.plot(point_2[0], point_2[1], "r+")
ax.annotate("Point_2_repos", (point_2[0], point_2[1]))

ax.plot(max_view_point[0], max_view_point[1], "r+")
ax.annotate("View_1", (max_view_point[0], max_view_point[1]))


plot_line(point_1, max_view_point)
plot_line(point_1, point_2)
plot_line(max_view_point, point_2)


vec1 = points_to_vec(point_1, max_view_point)
vec3 = points_to_vec(point_1, point_2)
vec2 = points_to_vec(max_view_point, point_2)

# Create a Rectangle patch
rect = patches.Rectangle((0,0),env_size,env_size,linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

angle = np.degrees(get_angle_vec_3(vec1, vec2, vec3))
angle_reset = angle + 90

print("angle = ", angle)
print("angle reset = ", angle_reset)

print("bin = ", angle_reset // 15)

ax.set_xlim(-100, 600)
ax.set_ylim(-100, 600)

plt.show()
