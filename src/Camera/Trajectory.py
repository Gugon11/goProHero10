class Trajectory:
    def __init__(self):
        # self.car_transform_list = []
        
        self.circles_pos_list = []
        self.length_of_each_segment = []
        self.segment_identifier = []
        self.type_of_curve = []
        self.index_type_of_curve = 0
        self.index_type_of_curve_short_distance = 0

        self.short_dist_index = 0
        self.last_short_dist_index = [0, 0, 0]

        self.next_circle_index = 0
        self.last_distance = 0.0
        self.point_car_on_trajectory = None
        self.point_car_on_trajectory_short_distance = None
        self.num_laps = 0
        self.new_lap = True

        self.FLAG_CUBES_V2 = True

    def shortest_point_to_curve(self, point1, point2, position):
        # Placeholder for the actual shortest point to curve V2 logic
        pass

    def shortest_point_to_line(self, point1, point2, position):
        # Placeholder for the actual shortest point to line logic
        pass

    def update_lap_number(self):
        # Placeholder for the actual update lap number logic
        pass

    def get_distance_to_origin(self):
        # Placeholder for the actual get distance to origin V2 logic
        pass
    
    def checkReward(self, car):
        minDist = float('inf')

        # Identify where the car is on the track
        for i in range(len(self.circles_pos_list)):
            if i == len(self.circles_pos_list) - 1:
                self.point_car_on_trajectory = self.shortest_point_to_curve(self.circles_pos_list[i], self.circles_pos_list[0], car.position)

                short_dist = self.point_car_on_trajectory - car.position
                if short_dist < minDist:
                    minDist = short_dist
                    self.short_dist_index = i
                    self.point_car_on_trajectory_short_distance = self.point_car_on_trajectory
                    self.index_type_of_curve_short_distance = len(self.type_of_curve) - 1
            
            else:
                if self.segment_identifier[i] == 1:  # if it's a straight
                    self.point_car_on_trajectory = self.shortest_point_to_line(self.circles_pos_list[i], self.circles_pos_list[i + 1], car.position)

                    short_dist = self.point_car_on_trajectory - car.position

                    if short_dist < min_dist:
                        min_dist = short_dist
                        self.short_dist_index = i
                        self.point_car_on_trajectory_short_distance = self.point_car_on_trajectory
                    
                    else:
                        self.point_car_on_trajectory = self.shortest_point_to_curve(self.circles_pos_list[i], self.circles_pos_list[0], car.position)

                        short_dist = self.point_car_on_trajectory - car.position
                        if short_dist < minDist:
                            minDist = short_dist
                            self.short_dist_index = i
                            self.point_car_on_trajectory_short_distance = self.point_car_on_trajectory
                            self.index_type_of_curve_short_distance = len(self.type_of_curve) - 1
        
        if self.last_short_dist_index[0] != self.short_dist_index:
            self.last_short_dist_index[2] = self.last_short_dist_index[1]
            self.last_short_dist_index[1] = self.last_short_dist_index[0]
            self.last_short_dist_index[0] = self.short_dist_index
        
        self.update_lap_number()

        dist_to_origin = self.get_distance_to_origin()

        if dist_to_origin <= self.last_distance:
            self.last_distance = dist_to_origin
            return False
        else:
            self.last_distance = dist_to_origin
            return True