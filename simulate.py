from map import *
from obstacle import *
from planner import *
import numpy as np

MCTNodeValue.action_list = [0, 1, 2, 3, 4, 5, 6]       # Six available actions in 2D world

seed = 0
robot = Robot([5.0, 50.0], 0.0, 0.0, seed)
sim_map = Map((80.0, 60.0), robot, [70.0, 50.0], 1.0, seed)

# Static; L_shape
# L_obs_1 = PolyObstacle([1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-30.0, 10.0, 20.0, -60.0])
# L_obs_2 = PolyObstacle([1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-20.0, 10.0, 10.0, -20.0])
# sim_map.add_obstacle(L_obs_1)
# sim_map.add_obstacle(L_obs_2)

# Static; Block
B_obs_1 = PolyObstacle([1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-40.0, 45.0, 35.0, -50.0])
B_obs_2 = PolyObstacle([1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-65.0, 45.0, 55.0, -51.5])
B_obs_3 = PolyObstacle([1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-80.0, -5.0, -5.0, -43.0])


sim_map.add_obstacle(B_obs_1)
sim_map.add_obstacle(B_obs_2)
sim_map.add_obstacle(B_obs_3)

# Static: Trap
# T_obs_1 = PolyObstacle([1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-65.0, 58.5, 55.0, -60.0])
# T_obs_2 = PolyObstacle([1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-65.0, 53.5, 55.0, -55.0])
# T_obs_3 = PolyObstacle([1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-65.0, 55.0, 63.5, -58.5])

# sim_map.add_obstacle(T_obs_1)
# sim_map.add_obstacle(T_obs_2)
# sim_map.add_obstacle(T_obs_3)

# Static: No Trap
NT_obs_1 = PolyObstacle([1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-65.0, 53.5, 55.0, -60.0])

sim_map.add_obstacle(NT_obs_1)

# Dynamic: Velocity Obstacle
x_trajectory = 10.0 * np.sin(np.linspace(0, np.pi, 100))
y_trajectory = np.zeros(x_trajectory.shape)
V_obs_1 = VelocityPolyObstacle(x_trajectory, y_trajectory, [1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-20.0, 45.0, 15.0, -50.0])
y_trajectory = 10.0 * np.sin(np.linspace(0, np.pi, 100))
x_trajectory = np.zeros(x_trajectory.shape)
V_obs_2 = VelocityPolyObstacle(x_trajectory, y_trajectory, [1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-15.0, 45.0, 10.0, -50.0])
y_trajectory = 5.0 - np.abs(5 - np.linspace(0, 10.0, 100))
x_trajectory = np.zeros(x_trajectory.shape)
V_obs_3 = VelocityPolyObstacle(x_trajectory, y_trajectory, [1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-45.0, 45.0, 40.0, -50.0])

sim_map.add_obstacle(V_obs_1)
sim_map.add_obstacle(V_obs_2)
sim_map.add_obstacle(V_obs_3)

# Dynamic: Deformable Obstacle
D_obs_1 = DeformablePolyObstacle(5.0, [1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-58.0, 51.5, 55.0, -53.5])
D_obs_2 = DeformablePolyObstacle(np.inf, [1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-58.0, 43.0, 55.0, -45.0])

sim_map.add_obstacle(D_obs_1)
sim_map.add_obstacle(D_obs_2)

planner = MCTS_Planner(sim_map)

sim_map.init_visualize()

while(True): 
    robot_action = planner.plan()
    if not sim_map.update(robot_action):    # Collision happened
        sys.exit('Collision detected!!!')
    sim_map.update_visualize()
