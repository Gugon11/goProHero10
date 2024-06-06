import numpy as np


class Trajectory:
    def __init__(self):
        
        
        self.circles_pos_list = []
        self.length_of_each_segment = []
        self.segment_identifier = [1, 0, 1, 0, 0, 0, 1, 0]
        self.type_of_curve = [1, 2, 2, 2, 3]
        self.index_type_of_curve = 0
        self.index_type_of_curve_short_distance = 0

        self.short_dist_index = 0
        self.last_short_dist_index = [0, 0, 0]

        self.next_circle_index = 0
        self.last_distance = 0.0
        self.point_car_on_trajectory = np.array([])
        self.point_car_on_trajectory_short_distance = np.array([])
        self.num_laps = 0
        self.new_lap = True

        self.FLAG_CUBES_V2 = True

    def unit_vector(self, vector):
        return vector/np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        angle_radians = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        angle_degrees = np.degrees(angle_radians)
        return angle_degrees


    def shortest_point_to_curve(self, curve_point1 = np.array([]), curve_point2 = np.array([]), car_position = np.array([])):
        center = np.array([0, 0, 0])
        final_point1 = np.array([0, 0, 0])
        final_point2 = np.array([0, 0, 0])
        v, vec1, vec2 = np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])
        closest_point_on_circle = np.array([0, 0, 0])

        if self.type_of_curve[self.index_type_of_curve] == 1:
            final_point1 = np.add(curve_point1, np.array([0, -100, 0]))
            final_point2 = np.add(curve_point2, np.array([-100, 0, 0]))
        
        elif self.type_of_curve[self.index_type_of_curve] == 2:
            center = np.array([(curve_point1[0] + curve_point2[0]) / 2,
                            (curve_point1[1] + curve_point2[1]) / 2,
                            (curve_point1[2] + curve_point2[2]) / 2])

            r = np.add(np.linalg.norm(curve_point1-center), np.linalg.norm(curve_point2-center)) / 2
            v = car_position - center
            vec1 = curve_point1 - center
            vec2 = curve_point2 - center
            start_angle = np.arctan2(vec1[0], vec1[1])
            end_angle = np.arctan2(vec2[0], vec2[1])
            car_angle = np.arctan2(v[0], v[1])

            if start_angle < 0:
                start_angle += 2 * np.pi
            if end_angle < 0:
                end_angle += 2 * np.pi
            if car_angle < 0:
                car_angle += 2 * np.pi

            if (car_angle >= start_angle and car_angle < end_angle and self.index_type_of_curve != 2) or \
            (not (car_angle >= start_angle and car_angle < end_angle) and self.index_type_of_curve == 2):
                closest_point_on_circle = center + (v / np.linalg.norm(v) * r)
            else:
                closest_point_on_circle = np.array([0, 0, 0])
            
            self.index_type_of_curve += 1
            if self.index_type_of_curve >= len(self.type_of_curve):
                self.index_type_of_curve = 0

            return closest_point_on_circle
        
        elif self.type_of_curve[self.index_type_of_curve] == 3:
            final_point1 = np.add(curve_point1, np.array([100, 0, 0]))
            final_point2 = np.add(curve_point2, np.array([0, -100, 0]))

        a1 = np.subtract(final_point1[0], curve_point1[0])
        b1 = np.subtract(curve_point1[1], final_point1[1])
        c1 = (a1 * curve_point1[1]) + (b1 * curve_point1[0])

        a2 = np.subtract(final_point2[0], curve_point2[0])
        b2 = np.subtract(curve_point2[1], final_point2[1])
        c2 = (a2 * curve_point2[1]) + (b2 * curve_point2[0])

        determinant = (a1 * b2) - (a2 * b1)

        y = ((b2 * c1) - (b1 * c2)) / determinant
        x = ((a1 * c2) - (a2 * c1)) / determinant

        center = np.array([x, y, curve_point1[2]])

        r = np.add(np.linalg.norm(curve_point1-center), np.linalg.norm(curve_point2-center)) / 2

        v = car_position - center
        vec1 = curve_point1 - center
        vec2 = curve_point2 - center

        start_angle = np.arctan2(vec1[0], vec1[1])
        end_angle = np.arctan2(vec2[0], vec2[1])
        car_angle = np.arctan2(v[0], v[1])

        if start_angle < 0:
            start_angle += 2 * np.pi
        if end_angle < 0:
            end_angle += 2 * np.pi
        if car_angle < 0:
            car_angle += 2 * np.pi

        if car_angle >= start_angle and car_angle < end_angle:
            closest_point_on_circle = center + (v / np.linalg.norm(v) * r)
        else:
            closest_point_on_circle = np.array([0, 0, 0])

        self.index_type_of_curve += 1
        if self.index_type_of_curve >= len(self.type_of_curve):
            self.index_type_of_curve = 0

        return closest_point_on_circle
        

    def shortest_point_to_line(self, line_point1 = np.array([]), line_point2 =np.array([]), car_position = np.array([])):
        AB = line_point2 - line_point1
        AC = car_position - line_point1

        t = np.dot(AC, AB) / np.dot(AB, AB)
        t = np.clip(t, 0, 1)

        closest_point_on_line = line_point1 + (AB * t)
        return closest_point_on_line
        

    def update_lap_number(self):
        arr = [0, 7, 6]
        reset_arr = [1, 0, 7]
        if self.new_lap and self.last_short_dist_index == arr:
            self.num_laps += 1
            self.new_lap = False
        if not self.new_lap and self.last_short_dist_index == reset_arr:
            self.new_lap = True

    def get_distance_to_trajectory(self, car_position = np.array([])):
        return np.linalg.norm(self.point_car_on_trajectory_short_distance - car_position)
    
    def checkReward(self, car_position = np.array([])):
        minDist = float('inf')

        # Identify where the car is on the track
        for i in range(len(self.circles_pos_list)):
            if i == len(self.circles_pos_list) - 1:
                self.point_car_on_trajectory = self.shortest_point_to_curve(self.circles_pos_list[i], self.circles_pos_list[0], car_position)

                short_dist = np.linalg.norm(self.point_car_on_trajectory - car_position)
                if short_dist < minDist:
                    minDist = short_dist
                    self.short_dist_index = i
                    self.point_car_on_trajectory_short_distance = self.point_car_on_trajectory
                    self.index_type_of_curve_short_distance = len(self.type_of_curve) - 1
            
            else:
                if self.segment_identifier[i] == 1:  # if it's a straight
                    self.point_car_on_trajectory = self.shortest_point_to_line(self.circles_pos_list[i], self.circles_pos_list[i + 1], car_position)

                    short_dist = np.linalg.norm(self.point_car_on_trajectory - car_position)

                    if short_dist < min_dist:
                        min_dist = short_dist
                        self.short_dist_index = i
                        self.point_car_on_trajectory_short_distance = self.point_car_on_trajectory
                    
                    else:
                        self.point_car_on_trajectory = self.shortest_point_to_curve(self.circles_pos_list[i], self.circles_pos_list[0], car_position)

                        short_dist = np.linalg.norm(self.point_car_on_trajectory - car_position)
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
        
    def get_distance_to_origin(self):
        total_distance = 0
        

        for _ in range(self.num_laps):
            for length in self.length_of_each_segment:
                total_distance += length

        for i in range(self.short_dist_index):
            total_distance += self.length_of_each_segment[i]

        if self.segment_identifier[self.short_dist_index] == 1:
            total_distance += self.circles_pos_list[self.short_dist_index] - self.point_car_on_trajectory_short_distance
        else:
            center = np.array([0, 0, 0])
            finalpoint1 = np.array([0, 0, 0])
            finalpoint2 = np.array([0, 0, 0])
            curve_point1 = np.array([0, 0, 0])
            curve_point2 = np.array([0, 0, 0])

            if self.short_dist_index == self.circles_pos_list - 1:
                curve_point1 = self.circles_pos_list[self.short_dist_index]
                curve_point2 = self.circles_pos_list[0]
            
            else:
                curve_point1 = self.circles_pos_list[self.short_dist_index]
                curve_point2 = self.circles_pos_list[self.short_dist_index+1]

            if self.type_of_curve[self.index_type_of_curve_short_distance] == 1:
                finalpoint1 = curve_point1 + np.array([0, -100, 0])
                finalpoint2 = curve_point2 + np.array([-100, 0, 0])
            
            elif self.type_of_curve[self.index_type_of_curve_short_distance] == 2:
                center = np.array([(curve_point1[0] + curve_point2[0]) / 2, (curve_point1[1] + curve_point2[1]) / 2, (curve_point1[2] + curve_point2[2]) / 2])
                r1 = (np.linalg.norm(curve_point1 - center) + np.linalg.norm(curve_point2 - center)) / 2
                v11 = curve_point1 - center
                v22 = self.point_car_on_trajectory_short_distance - center
                perimeter1 = 2 * np.pi * r1 * (self.angle_between(v11, v22) / 360)
                total_distance += perimeter1
                return total_distance
            
            elif self.type_of_curve[self.index_type_of_curve_short_distance] == 3:
                finalpoint1 = curve_point1 + np.array([100, 0, 0])
                finalpoint2 = curve_point2 + np.array([0, -100, 0])

            a1 = finalpoint1[0] - curve_point1[0]
            b1 = curve_point1[1] - finalpoint1[1]
            c1 = (a1 * curve_point1[1]) + (b1 * curve_point1[0])

            a2 = finalpoint2[0] - curve_point2[0]
            b2 = curve_point2[1] - finalpoint2[1]
            c2 = (a2 * curve_point2[1]) + (b2 * curve_point2[0])

            determinant = (a1 * b2) - (a2 * b1)
            y = ((b2 * c1) - (b1 * c2)) / determinant
            x = ((a1 * c2) - (a2 * c1)) / determinant

            center = np.array([x, y, curve_point1[2]])
            r = (np.linalg.norm(curve_point1 - center) + np.linalg.norm(curve_point2 - center)) / 2
            v1 = curve_point1 - center
            v2 = self.point_car_on_trajectory_short_distance - center
            perimeter = 2 * np.pi * r * (self.angle_between(v1, v2) / 360)
            total_distance += perimeter

        return total_distance

    def get_length_of_each_segment(self):
        for i in range(len(self.circles_pos_list)):
            if i == len(self.circles_pos_list) - 1:
                curve_point1 = self.circles_pos_list[i]
                curve_point2 = self.circles_pos_list[0]

                final_point1 = curve_point1 + np.array([100, 0, 0])
                final_point2 = curve_point2 + np.array([0, -100, 0])

                a1 = final_point1[0] - curve_point1[0]
                b1 = curve_point1[1] - final_point1[1]
                c1 = (a1 * curve_point1[1]) + (b1 * curve_point1[0])

                a2 = final_point2[0] - curve_point2[0]
                b2 = curve_point2[1] - final_point2[1]
                c2 = (a2 * curve_point2[1]) + (b2 * curve_point2[0])

                determinant = (a1 * b2) - (a2 * b1)

                y = ((b2 * c1) - (b1 * c2)) / determinant
                x = ((a1 * c2) - (a2 * c1)) / determinant

                center = np.array([x, y, curve_point1[2]])
                r = (np.linalg.norm(curve_point1 - center)) / 2 + (np.linalg.norm(curve_point2 - center)) / 2

                perimeter = 2 * np.pi * r / 4  # 90 degree turns
                self.lenght_of_each_segment.append(perimeter)

                self.index_type_of_curve += 1
                if self.index_type_of_curve >= len(self.type_of_curve):
                    self.index_type_of_curve = 0
            else:
                if self.segment_identifier[i] == 1:  # if straight
                    self.lenght_of_each_segment.append((np.linalg.norm(self.circles_pos_list[i] - self.circles_pos_list[i + 1])))
                else:  # if curve
                    curve_point1 = self.circles_pos_list_pos_list[i]
                    curve_point2 = self.circles_pos_list_pos_list[i + 1]

                    if self.type_of_curve[self.index_type_of_curve] == 1:
                        final_point1 = curve_point1 + np.array([0, -100, 0])
                        final_point2 = curve_point2 + np.array([-100, 0, 0])

                        a1 = final_point1[0] - curve_point1[0]
                        b1 = curve_point1[1] - final_point1[1]
                        c1 = (a1 * curve_point1[1]) + (b1 * curve_point1[0])

                        a2 = final_point2[0] - curve_point2[0]
                        b2 = curve_point2[1] - final_point2[1]
                        c2 = (a2 * curve_point2[1]) + (b2 * curve_point2[0])

                        determinant = (a1 * b2) - (a2 * b1)

                        y = ((b2 * c1) - (b1 * c2)) / determinant
                        x = ((a1 * c2) - (a2 * c1)) / determinant

                        center = np.array([x, y, curve_point1[2]])
                        r = (np.linalg.norm(curve_point1 - center)) / 2 + (np.linalg.norm(curve_point2 - center)) / 2

                        perimeter = 2 * np.pi * r / 4  # 90 degree turns
                        self.lenght_of_each_segment.append(perimeter)
                    elif self.type_of_curve[self.index_type_of_curve] == 2:
                        center = (curve_point1 + curve_point2) / 2
                        r = (np.linalg.norm(curve_point1 - center)) / 2 + (np.linalg.norm(curve_point2 - center)) / 2

                        perimeter = 2 * np.pi * r / 2  # 180 degree turns
                        self.lenght_of_each_segment.append(perimeter)

                    self.index_type_of_curve += 1
                    if self.index_type_of_curve >= len(self.type_of_curve):
                        self.index_type_of_curve = 0

        self.index_type_of_curve = 0  # make sure the index_type_of_curve is reset

    def get_sign_of_distance_to_trajectory_v2(self, car_pos = np.array([])):
        sign = 1.0
        if self.short_dist_index == len(self.circles_pos_list) - 1:
            curve_point1 = self.circles_pos_list[self.short_dist_index]
            curve_point2 = self.circles_pos_list[0]

            final_point1 = curve_point1 + np.array([100, 0, 0])
            final_point2 = curve_point2 + np.array([0, -100, 0])

            a1 = final_point1[0] - curve_point1[0]
            b1 = curve_point1[1] - final_point1[1]
            c1 = a1 * curve_point1[1] + b1 * curve_point1[0]

            a2 = final_point2[0] - curve_point2[0]
            b2 = curve_point2[1] - final_point2[1]
            c2 = a2 * curve_point2[1] + b2 * curve_point2[0]

            determinant = a1 * b2 - a2 * b1

            y = (b2 * c1 - b1 * c2) / determinant
            x = (a1 * c2 - a2 * c1) / determinant

            center = np.array([x, y, curve_point1[2]])
            r = (np.linalg.norm(center - curve_point1)) / 2 + (np.linalg.norm(center - curve_point2)) / 2

            dist = (np.linalg.norm(center - car_pos)) - r
            sign = 1.0 if dist >= 0 else -1.0

            return sign
        else:
            if self.segment_identifier[self.short_dist_index] == 1:
                delta = 0
                if self.short_dist_index == 0:
                    y = (self.circles_pos_list[self.short_dist_index][1] + self.circles_pos_list[self.short_dist_index + 1][1]) / 2
                    delta = abs(y) - abs(car_pos[1])
                elif self.short_dist_index == 2:
                    x = (self.circles_pos_list[self.short_dist_index][0] + self.circles_pos_list[self.short_dist_index + 1][0]) / 2
                    delta = abs(x) - abs(car_pos[0])
                elif self.short_dist_index == 6:
                    x = (self.circles_pos_list[self.short_dist_index][0] + self.circles_pos_list[self.short_dist_index + 1][0]) / 2
                    delta = abs(car_pos[0]) - abs(x)

                sign = 1.0 if delta >= 0 else -1.0

                return sign
            else:
                curve_point1 = self.circles_pos_list[self.short_dist_index]
                curve_point2 = self.circles_pos_list[self.short_dist_index + 1]

                if self.type_of_curve[self.index_type_of_curve_short_distance] == 1:
                    final_point1 = curve_point1 + np.array([0, -100, 0])
                    final_point2 = curve_point2 + np.array([-100, 0, 0])
                elif self.type_of_curve[self.index_type_of_curve_short_distance] == 2:
                    center = (curve_point1 + curve_point2) / 2
                    r1 = (np.linalg.norm(center - curve_point1)) / 2 + (np.linalg.norm(center - curve_point2)) / 2

                    dist = (np.linalg.norm(center - car_pos)) - r1

                    if self.short_dist_index == 4:
                        sign = -1.0 if dist >= 0 else 1.0
                    else:
                        sign = 1.0 if dist >= 0 else -1.0

                    return sign
                elif self.type_of_curve[self.index_type_of_curve_short_distance] == 3:
                    final_point1 = curve_point1 + np.array([100, 0, 0])
                    final_point2 = curve_point2 + np.array([0, -100, 0])

                a1 = final_point1[0] - curve_point1[0]
                b1 = curve_point1[1] - final_point1[1]
                c1 = a1 * curve_point1[1] + b1 * curve_point1[0]

                a2 = final_point2[0] - curve_point2[0]
                b2 = curve_point2[1] - final_point2[1]
                c2 = (a2 * curve_point2[1]) + (b2 * curve_point2[0])

                determinant = (a1 * b2) - (a2 * b1)

                y = ((b2 * c1) - (b1 * c2)) / determinant
                x = ((a1 * c2) - (a2 * c1)) / determinant

                center = np.array([x, y, curve_point1[2]])
                r = (np.linalg.norm(curve_point1 - center)) / 2 + (np.linalg.norm(curve_point2 - center)) / 2

                dist = (np.linalg.norm(center - car_pos)) - r
                sign = 1.0 if dist >= 0 else -1.0

                return sign
