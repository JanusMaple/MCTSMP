import numpy as np
import random

class Robot: 
    inc_vel = 0.1
    inc_a_vel = 0.3
    inc_a_vel_min = 0.1
    state_list_len = 4                      # Number of float type numbers needed to represent a state

    def __init__(self, init_pos=[], init_vel=0.0, ini_a_vel=0.0, seed=0) -> None: 
        self.pos = init_pos
        self.vel = init_vel
        self.a_vel = ini_a_vel
        self.seed = seed
        self.e_brake = False
        random.seed(seed)

    def pull_e_brake(self): 
        self.e_brake = True

    def get_state_as_array(self):           # Return a column vector containing useful current robot state info
        return np.array([self.pos[0], self.pos[1], self.vel, self.a_vel]).transpose()

    def apply_random_action(self): 
        self.apply_action(random.randint(0,5))

    def apply_action(self, action_index):   # six actions: [0 1 2] for velocity change and [3 4 5] for angular velocity change
        
        assert (0 <= action_index) & (action_index <= 6)
        if action_index <= 2: 
            delta = (action_index - 1.0) * self.inc_vel
            self.vel = np.max([0.0, self.vel + delta])
        elif action_index <= 5: 
            delta = (action_index - 4.0) * np.max([self.inc_a_vel_min, self.inc_a_vel - self.vel * (self.inc_a_vel_min - self.inc_a_vel)])
            self.a_vel = self.a_vel + delta
        elif self.vel == 0.0: 
            self.direction = self.a_vel + np.pi

    def update(self): 
        if self.e_brake: 
            self.vel = 0.0
            self.avel = 0.0
        self.pos[0] = self.pos[0] + np.cos(self.a_vel) * self.vel
        self.pos[1] = self.pos[1] + np.sin(self.a_vel) * self.vel
