import numpy as np

class DataProcessing:
    def __init__(self,df):
        self.df = df
    def add_speed(self):
        self.df = self.df.assign(speed =np.sqrt(self.df.transmitter_x_velocity**2 + self.df.transmitter_y_velocity**2))
    def add_distance(self):
        self.df = self.df.assign(distance =DataProcessing.ecu_distance([self.df.transmitter_x_position,self.df.transmitter_y_position],[self.df.receiver_x_position,self.df.receiver_y_position]))
    def ecu_distance(x,y): # Euclidean Distance (emitter, receiver)
        return np.sqrt(((x[0]-y[0])**2+(x[1]-y[1])**2))