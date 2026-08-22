from map import *
from obstacle import *
from planner import *
import numpy as np
from tqdm import tqdm
import time
import json

MCTNodeValue.action_list = [0, 1, 2, 3, 4, 5, 6]       # Six available actions in 2D world

# Static; Block
B_obs_1 = PolyObstacle([1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-40.0, 2.0, 35.0, -7.0])
B_obs_2 = PolyObstacle([1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-65.0, 2.0, 55.0, -8.5])

# Static: No Trap
NT_obs_1 = PolyObstacle([1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-65.0, 10.5, 55.0, -17.0])

# Dynamic: Velocity Obstacle
x_trajectory = 10.0 * np.sin(np.linspace(0, np.pi, 100))
y_trajectory = np.zeros(x_trajectory.shape)
V_obs_1 = VelocityPolyObstacle(x_trajectory, y_trajectory, [1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-20.0, 2.0, 15.0, -7.0])
y_trajectory = 10.0 * np.sin(np.linspace(0, np.pi, 100))
x_trajectory = np.zeros(x_trajectory.shape)
V_obs_2 = VelocityPolyObstacle(x_trajectory, y_trajectory, [1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-15.0, 2.0, 10.0, -7.0])
y_trajectory = 5.0 - np.abs(5 - np.linspace(0, 10.0, 100))
x_trajectory = np.zeros(x_trajectory.shape)
V_obs_3 = VelocityPolyObstacle(x_trajectory, y_trajectory, [1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-45.0, 2.0, 40.0, -7.0])

# Dynamic: Deformable Obstacle
D_obs_1 = DeformablePolyObstacle(5.0, [1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-58.0, 8.5, 55.0, -10.5])
D_obs_2 = DeformablePolyObstacle(np.inf, [1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-58.0, 0.0, 55.0, -2.0])

def sim_once(visualize): 
    seed = 0
    robot = Robot([5.0, 7.0], 0.0, 0.0, seed)
    sim_map = Map((80.0, 17.0), robot, [70.0, 7.0], 1.0, seed)
    sim_map.add_obstacle(B_obs_1)
    sim_map.add_obstacle(B_obs_2)
    sim_map.add_obstacle(NT_obs_1)
    sim_map.add_obstacle(V_obs_1)
    sim_map.add_obstacle(V_obs_2)
    sim_map.add_obstacle(V_obs_3)
    sim_map.add_obstacle(D_obs_1)
    sim_map.add_obstacle(D_obs_2)

    planner = MCTS_Planner(sim_map)

    if visualize: 
        sim_map.init_visualize()

    start_time = time.time()

    while(True): 
        cur_time = time.time()
        if cur_time - start_time >= 60.0: 
            return False, 60.0
        robot_action = planner.plan()
        if not sim_map.update(robot_action): 
            plt.close()
            return False, 0.0
        if visualize: 
            sim_map.update_visualize()
        if sim_map.goal_reached: 
            plt.close()
            return True, cur_time - start_time

visualize = False
success_rate = 0.0
finish_time = []
for i in tqdm(range(50)): 
    suc, sim_time = sim_once(visualize)
    finish_time.append(sim_time)
    if suc: 
        success_rate = (success_rate * i + 1) / (i + 1)
    else: 
        success_rate = (success_rate * i) / (i + 1)

data = {"success_rate": success_rate, "time": finish_time}
print(data)
with open("sim_result.json", "w") as file:
    json.dump(data, file)
