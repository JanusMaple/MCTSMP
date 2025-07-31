from robot import *
from obstacle import *
import matplotlib.pyplot as plt
import numpy as np
import sys
from copy import deepcopy

class Map: 
    def __init__(self, size, robot, goal, goal_radius, seed) -> None: 
        self.goal = goal
        self.robot = robot
        self.size = size                # tuple: (width, height)
        self.seed = seed
        border_obstacle = PolyObstacle([1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-size[0], 0.0, 0.0, -size[1]], False)
        self.obstacles = [border_obstacle]
        self.figure = None
        self.robot_plot = None
        self.escape_plot = None
        self.obs_plot = []

        self.local_trap = []
        self.ori_vel = 0
        self.stopped_time = 0
        self.trapped = False

        self.goal_reached = False
        self.escape_goal = self.goal
        self.goal_radius = goal_radius
        self.default_gray_level = 0.7
        
    def add_obstacle(self, obstacle): 
        self.obstacles.append(deepcopy(obstacle))

    def update_environment_predictively(self, time, state_list): 
        for o in self.obstacles: 
            if o.is_dynamic: 
                if isinstance(o, VelocityPolyObstacle): 
                    o.update(time)
                if isinstance(o, DeformablePolyObstacle): 
                    o.update_based_on_history(state_list[0:2, :])

    def update(self, robot_action): 
        # Move Robot
        if robot_action is None: 
            self.robot.apply_random_action()
        else:
            self.robot.apply_action(robot_action)
        ori_pos = deepcopy(self.robot.pos)
        self.robot.update()

        # Spawn Local Minima Repulsion Point if Needed
        if self.ori_vel == 0.0 and self.robot.vel == 0.0: 
            self.stopped_time = self.stopped_time + 1
        else: 
            self.stopped_time = 0
        
        self.trapped = False
        dis = np.sqrt((self.goal[0] - self.robot.pos[0]) ** 2 + (self.goal[1] - self.robot.pos[1]) ** 2)
        if dis <= self.goal_radius and self.robot.vel == 0.0: 
            self.goal_reached = True
        if dis > self.goal_radius and self.stopped_time >= 3: 
            self.trapped = True
            cloesd_dis = np.inf
            cloesd_line = [0.0, 0.0, 0.0]
            for o in self.obstacles: 
                if isinstance(o, DeformablePolyObstacle) and not o.bad_to_manipulate: 
                    continue
                else: 
                    (temp_dis, temp_line) = o.distance(self.robot.pos[0], self.robot.pos[1])
                if temp_dis <= cloesd_dis: 
                    cloesd_dis = temp_dis
                    cloesd_line = temp_line
            [a, b, c] = cloesd_line
            [p, q] = self.robot.pos
            x = (b ** 2 * p - a * b * q - a * c) / (a ** 2 + b ** 2)
            y = (a ** 2 * q - a * b * p - b * c) / (a ** 2 + b ** 2)
            self.local_trap.append((self.robot.pos + np.array([x, y])) / 2.0)
        elif dis <= self.goal_radius: 
            self.local_trap = []
        else: 
            i = 0
            while i < len(self.local_trap): 
                lt = self.local_trap[i]
                dis = np.sqrt((lt[0] - self.robot.pos[0]) ** 2 + (lt[1] - self.robot.pos[1]) ** 2)
                if dis >= self.goal_radius: 
                    self.local_trap.pop(i)
                else: 
                    i = i + 1
        self.ori_vel = self.robot.vel

        # Update Obstacles
        for o in self.obstacles: 
            if isinstance(o, DeformablePolyObstacle): 
                mark_escape_loc = False
                if not o.bad_to_manipulate and o.deform_rate() == 0.0: 
                    mark_escape_loc = True
                o.update(self.robot.pos[0], self.robot.pos[1])
                if mark_escape_loc and o.deform_rate() > 0.0: 
                    self.escape_goal = (np.array(ori_pos) - self.robot.pos) * 2.0 + self.robot.pos
                if o.is_dangerous(): 
                    o.bad_to_manipulate = True
            else: 
                o.update()
            if o.collision_detect(self.robot.pos[0], self.robot.pos[1]): 
                # sys.exit('Collision detected!!!')
                return False
        return True

    def explored(self): 
        xi = min(self.exploration_map.shape[0] - 1, int(np.floor(self.robot.pos[0] / self.explore_map_res)))
        yi = min(self.exploration_map.shape[1] - 1, int(np.floor(self.robot.pos[1] / self.explore_map_res)))
        self.exploration_map[xi, yi] = self.exploration_map[xi, yi] * 0.5

    def explore_score(self, x, y): 
        xi = min(self.exploration_map.shape[0] - 1, int(np.floor(x / self.explore_map_res)))
        yi = min(self.exploration_map.shape[1] - 1, int(np.floor(y / self.explore_map_res)))
        return self.exploration_map[xi, yi]

    def init_visualize(self): 
        plt.ion()
        self.figure = plt.figure(figsize=(10.0 * self.size[0] / max(self.size), 10.0 * self.size[1] / max(self.size)))
        plt.axis([0.0, self.size[0], 0.0, self.size[1]])
        plt.gca().set_aspect('equal')
        plt.xlim(0.0, self.size[0])
        plt.ylim(0.0, self.size[1])
        plt.xticks([])
        plt.yticks([])

        plt.gca().add_patch(plt.Circle((self.goal[0], self.goal[1]), self.goal_radius, color='r'))
        # self.escape_plot, = plt.plot(self.escape_goal[0], self.escape_goal[1], 'ob', zorder=5)
        self.robot_plot, = plt.plot(self.robot.pos[0], self.robot.pos[1], 'k.',zorder=5)
        for i in range(1, len(self.obstacles)): 
            if isinstance(self.obstacles[i], DeformablePolyObstacle): 
                vel_obs_plot, = plt.fill(self.obstacles[i].x, self.obstacles[i].y, facecolor=(self.default_gray_level, self.default_gray_level, self.default_gray_level))
            else: 
                vel_obs_plot, = plt.fill(self.obstacles[i].x, self.obstacles[i].y, facecolor='black', zorder=3)
            if self.obstacles[i].is_dynamic: 
                self.obs_plot.append(vel_obs_plot)
            

    def update_visualize(self): 
        self.robot_plot.set_xdata(self.robot.pos[0])
        self.robot_plot.set_ydata(self.robot.pos[1])
        # self.escape_plot.set_xdata(self.escape_goal[0])
        # self.escape_plot.set_ydata(self.escape_goal[1])
        j = 0
        for i in range(1, len(self.obstacles)): 
            if self.obstacles[i].is_dynamic: 
                self.obs_plot[j].set_xy(np.array([self.obstacles[i].x, self.obstacles[i].y]).T)
                if isinstance(self.obstacles[i], DeformablePolyObstacle): 
                    b =  self.default_gray_level * (1 - self.obstacles[i].deform_rate())
                    self.obs_plot[j].set_color((b, b, b))
                j = j + 1
        
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

class FakeMap(Map):                     # For evaluation ONLY (during a simulation of MCTS) without GUI
    def __init__(self, map) -> None:
        self.obstacles = deepcopy(map.obstacles)
        self.goal = map.goal
        self.goal_radius = map.goal_radius
        self.local_trap = map.local_trap
        self.trapped = map.trapped
        self.escape_goal = map.escape_goal

    def evaluate(self, robot_states=None): 
        if robot_states is None: 
            robot_states = np.expand_dims(self.robot.get_state_as_array(), -1)
        goal = self.goal
        score = 0.0
        x = robot_states[0, -1]
        y = robot_states[1, -1]
        v = robot_states[2, -1]
        w = robot_states[3, -1]
        vx = v * np.cos(w)
        vy = v * np.sin(w)
        escaping = False

        # Obstacle Avoidance Score
        dis_obs = np.inf
        for obs in self.obstacles: 
            if isinstance(obs, DeformablePolyObstacle):
                    if not obs.bad_to_manipulate:  
                        continue
                    else: 
                        if obs.deform_rate() > 0.0: 
                            goal = self.escape_goal
                            escaping = True
            (dis, line_function) = obs.distance(x, y)
            if dis < dis_obs: 
                dis_obs = dis
        
        if dis_obs <= 0: 
            score = score - 1.0
        elif dis_obs <= 0.5: 
            score = score - 0.5

        # Goal Navigation Score
        dis = np.sqrt((goal[0] - x) ** 2 + (goal[1] - y) ** 2)

        if v >= 0.75: 
            score = score - 1.0

        if not escaping and dis <= self.goal_radius and v == 0.0: 
            score = score + 1.0

        score = score + max(0.0, dis / 100.0 * ((goal[0] - x) / (0.1 + dis) * vx + (goal[1] - y) / (0.1 + dis) * vy))

        if escaping: 
            score = score + ((goal[0] - x) * vx + (goal[1] - y) * vy) / (dis + 0.01)

        # Local Minima Trap Score
        if self.trapped and v == 0.0: 
            score = score - 2.0

        if not self.trapped: 
            score = score + 2.0 / (1.0 + dis)
        else: 
            score = score + 0.2 / (1.0 + dis)

        if dis > self.goal_radius: 
            for lt in self.local_trap: 
                dis = np.sqrt((lt[0] - x) ** 2 + (lt[1] - y) ** 2)
                score = score - 2.0 / (1.0 + dis)

        return score