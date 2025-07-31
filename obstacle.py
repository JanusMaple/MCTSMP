import numpy as np

class PolyObstacle: 
    is_dynamic = False
    def __init__(self, a, b, c, inside=True, seed=0) -> None:
        np.random.seed(seed)
        self.a = a
        self.b = b
        self.c = c                                                                      # Polygon border a * x + b * y + c >= 0.0 in clockwise sequence
        self.x = []
        self.y = []
        self.inside = inside                                                            # If false, the obstacle is the outer world of the map
        self.update_vertex()
    
    def update_vertex(self): 
        l = len(self.a)
        for i in range(0, l): 
            a1 = self.a[i]
            b1 = self.b[i]
            c1 = self.c[i]
            a2 = self.a[(i + 1) % l]
            b2 = self.b[(i + 1) % l]
            c2 = self.c[(i + 1) % l]
            self.x.append((c2 * b1 - c1 * b2) / (a1 * b2 - a2 * b1))
            self.y.append((c1 * a2 - c2 * a1) / (a1 * b2 - a2 * b1))
        self.x.append(self.x[0])
        self.y.append(self.y[0])

    def update(self): 
        pass

    def collision_detect_base(self, x, y, x_bias=0.0, y_bias=0.0): 
        for a, b, c in zip(self.a, self.b, self.c):                                     # TO DO: Use matrix multiplication instead of using for loop
            if a * (x - x_bias) + b * (y - y_bias) + c > 0.0:                           # Out side of the polygon
                return not self.inside                                                  # No collision if inside of the polygon is obstacle
        return self.inside
    
    def distance_base(self, x, y, x_bias=0.0, y_bias=0.0): 
        argmin_para = []
        if self.inside: 
            dis = 0.0
            for a, b, c in zip(self.a, self.b, self.c): 
                dis_edge = a * (x - x_bias) + b * (y - y_bias) + c
                if dis_edge >= dis: 
                    argmin_para = [a, b, c - a * x_bias - b * y_bias]
                    dis = dis_edge
        else: 
            dis = np.inf
            for a, b, c in zip(self.a, self.b, self.c): 
                dis_edge = -(a * (x - x_bias) + b * (y - y_bias) + c)
                if dis_edge <= dis: 
                    argmin_para = [a, b, c - a * x_bias - b * y_bias]
                    dis = dis_edge
        return dis, argmin_para

    def collision_detect(self, x, y):                                                   # True: there is collision
        return self.collision_detect_base(x, y)
    
    def distance(self, x, y):                                                           # Minimum distance between the point and the polycon
        return self.distance_base(x, y)

    
class VelocityPolyObstacle(PolyObstacle):                                               # Abusing the notion here, actually meaning obstacles with velocity
    is_dynamic = True
    def __init__(self, x_trajectory, y_trajectory, a, b, c, inside=True, seed=0) -> None:
        PolyObstacle.__init__(self, a, b, c, inside, seed)                              # update_vertex() called and all vertex coordinates updated
        self.ori_x = self.x
        self.ori_y = self.y
        self.x_trajectory = x_trajectory
        self.y_trajectory = y_trajectory
        self.x_bias = 0.0
        self.y_bias = 0.0
        self.instant_vel = (0.0, 0.0)
        self.time = 0

    def collision_detect(self, x, y):                                                   # This only difference is the shifting of x and y
        return self.collision_detect_base(x, y, self.x_bias, self.y_bias)

    def update(self, delta_time = 1): 
        self.time = self.time + delta_time
        if self.time >= len(self.x_trajectory): 
            self.time = 0
        self.instant_vel = (self.x_trajectory[self.time] -self.x_bias, self.y_trajectory[self.time] -self.y_bias)
        self.x_bias = self.x_trajectory[self.time] + np.random.normal(0, 0.1) * delta_time
        self.y_bias = self.y_trajectory[self.time] + np.random.normal(0, 0.1) * delta_time
        self.x = self.ori_x + self.x_bias
        self.y = self.ori_y + self.y_bias
    
    def distance(self, x, y):                                                           # Minimum distance between the point and the polycon
        return self.distance_base(x, y, self.x_bias, self.y_bias)
    
class DeformablePolyObstacle(PolyObstacle):                                             # WARNING: A naive model for rectagular deformable obastcle that can be pushed in positive x-drection
    is_dynamic = True
    danger_factor = 0.7
    max_recover_step = 0.3
    interaction_dis = 0.001
    bad_to_manipulate = False
    def __init__(self, x_limit, a, b, c, inside=True, seed=0) -> None:
        super().__init__(a, b, c, inside, seed)
        self.x_bias = 0.0
        self.x_limit = x_limit

        self.y_range = [min(self.y), max(self.y)]
        self.ori_x = np.array(self.x)
        self.x_anchor = min(self.x)

    def collision_detect(self, x, y):
        if self.x_bias > self.x_limit: 
            return True
        return super().collision_detect_base(x, y, self.x_bias)

    def update_based_on_history(self, r_pos_list): 
        for i in range(r_pos_list.shape[1]): 
            r_x = r_pos_list[0, i]
            r_y = r_pos_list[1, i]
            self.update(r_x, r_y)

    def update(self, r_x, r_y): 
        if not self.collision_detect(r_x, r_y): 
            if r_y > self.y_range[1] or r_y < self.y_range[0]: 
                self.x_bias = max(0.0, self.x_bias - self.max_recover_step)
            else: 
                self.x_bias = max(r_x - self.x_anchor + self.interaction_dis, self.x_bias - self.max_recover_step, 0.0)
        else: 
            self.x_bias = min(r_x - self.x_anchor + self.interaction_dis, self.x_limit) # The simplest update rule
        self.x = self.ori_x + self.x_bias

    def is_dangerous(self): 
        return (self.deform_rate() >= self.danger_factor)
    
    def danger_rate(self): 
        return self.deform_rate / self.danger_factor
    
    def deform_rate(self): 
        return (self.x_bias / self.x_limit)
    
    def distance(self, x, y):                                                           # Minimum distance between the point and the polycon
        return self.distance_base(x, y, self.x_bias, 0.0)