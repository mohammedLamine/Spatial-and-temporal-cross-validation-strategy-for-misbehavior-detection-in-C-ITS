import numpy as np
import pandas as pd

from V2XMD  import data_processing

class PlausibilityChecks:
    def __init__(self,df):
        if (df.label.dtype ==np.int64):
            self.df=df.assign(truth =df.label!=1)
            return

        self.df=df.assign(truth =df.label!='Genuine')

    def art(self, threshold): # Acceptance Range Threshold (ART)
        self.df = self.df.assign(ART = self.df.distance>threshold)
        return self.df.assign(pred = self.df.distance>threshold)
    
    def saw(self,threshold): # Sudden Appereance Warning (SAW)
        result = self.df.assign(pred=self.df.sort_values('reception_time').groupby(['dataset','receiver_id','transmitter_id']).head(1)['distance']<threshold)
        result=result.sort_values('reception_time').groupby(['dataset','receiver_id','transmitter_id']).ffill()
        result.pred=result.pred.astype(bool)
        return result
    
    def sawBis(self,threshold): # Sudden Appereance Warning (SAW)
        self.df = self.df.assign(SAW=self.df.sort_values('reception_time').groupby(['dataset','receiver_id','transmitter_id']).head(1)['distance']<threshold)
        self.df.SAW.fillna(False,inplace= True)
    
    def ssc(self,threshold): # Simple Speed Check (SSC)
        shifted_values = self.df.sort_values('reception_time').groupby(['dataset','receiver_id','transmitter_id'])[['transmitter_x_position','transmitter_y_position','reception_time']].shift()
        shifted_values = self.df.assign(shifted_transmitter_x_position = shifted_values['transmitter_x_position'],
                                        shifted_transmitter_y_position=shifted_values['transmitter_y_position'],
                                        shifted_reception_time=shifted_values['reception_time'])
        shifted_values = shifted_values.assign(shift_distance = data_processing.DataProcessing.ecu_distance([shifted_values.transmitter_x_position,shifted_values.transmitter_y_position],[shifted_values.shifted_transmitter_x_position,shifted_values.shifted_transmitter_y_position]),
                                               elapsed_time   = abs(shifted_values.reception_time-shifted_values.shifted_reception_time)
                                              )
        shifted_values = shifted_values.assign(computed_speed   = shifted_values.shift_distance/shifted_values.elapsed_time)
        shifted_values = shifted_values.assign(speed_difference = abs(shifted_values.speed-shifted_values.computed_speed))
        shifted_values = shifted_values.assign(pred             = shifted_values.speed_difference>threshold)
        return shifted_values

    def sscBis(self,threshold): # Simple Speed Check (SSC)
        shifted_values = self.df.sort_values('reception_time').groupby(['dataset','receiver_id','transmitter_id'])[['transmitter_x_position','transmitter_y_position','reception_time']].shift()
        shifted_values = self.df.assign(shifted_transmitter_x_position = shifted_values['transmitter_x_position'],
                                        shifted_transmitter_y_position=shifted_values['transmitter_y_position'],
                                        shifted_reception_time=shifted_values['reception_time'])
        shifted_values = shifted_values.assign(shift_distance = data_processing.DataProcessing.ecu_distance([shifted_values.transmitter_x_position,shifted_values.transmitter_y_position],[shifted_values.shifted_transmitter_x_position,shifted_values.shifted_transmitter_y_position]),
                                               elapsed_time   = abs(shifted_values.reception_time-shifted_values.shifted_reception_time)
                                              )
        shifted_values = shifted_values.assign(computed_speed   = shifted_values.shift_distance/shifted_values.elapsed_time)
        shifted_values = shifted_values.assign(speed_difference = abs(shifted_values.speed-shifted_values.computed_speed))
        self.df        = self.df.assign(SSC = shifted_values.speed_difference>threshold)        

    def add_feature_vectors(self):
        sender_ID      = np.unique(np.array(self.df.iloc[:,7])) # Total number of unique emitters
        number_id_tr_s = len(sender_ID) 
        for i in range(number_id_tr_s): # For each unique emitter
            this             = self.df.loc[(self.df['transmitter_id'] == sender_ID[i])] # All messages sent by this emitter
            this_recevier_ID = np.unique(np.array(this.iloc[:,2])) # All receivers messaged by the emitter 
            number_id_re_s   = len(this_recevier_ID) # All unique receivers ID
            for j in range(number_id_re_s): # For each unique receiver
                b = this.loc[this['receiver_id'] == this_recevier_ID[j]]
                feature_1 = self.location_plausibility(b)
                feature_2 = self.movement_plausibility(b)
                feature_3,feature_4,feature_5,feature_6 = self.quantititative_information(b)
                #feature_7 = distance_check(b, 800)
                
                b = b.head(1) #?
                
                b['feature_1'] = feature_1
                b['feature_2'] = feature_2
                b['feature_3'] = feature_3
                b['feature_4'] = feature_4
                b['feature_5'] = feature_5
                b['feature_6'] = feature_6
                #b['feature_7'] = feature_7
                
                if i==0 and j==0:
                   feature_vector = b
                else:
                  feature_vector = pd.concat([feature_vector, b])
        return feature_vector
        
    def location_plausibility(self,b):
        # See Steven So's paper
        x_95 = [- 5.6983,  5.2265] 
        x_99 = [- 7.1795,  7.7077]
        y_95 = [- 8.1203,  8.0501]
        y_99 = [-12.1629, 12.0927]    
        ## score represents ... ?
        #  score  = [] # score is ... ?
        #  score.append(2) # For the start of the series, we think it is two  (Why?)
        score  = [2] # score value is 2 because....
        length = b.receiver_x_position.shape[0]  # Number of rows in data frame b #length(b)
        if length <=1: # if there is only one transmission between the emitter & the sender.
            return score
        for k in range(length-1):
            time_interval = (b.iloc[k+1]['reception_time'] - b.iloc[k]['reception_time'])
            x_pre_95_low = b.iloc[k]['transmitter_x_position'] + time_interval * (b.iloc[k]['transmitter_x_velocity'] +  x_95 [0] * time_interval * 0.1)
            x_pre_95_up  = b.iloc[k]['transmitter_x_position'] + time_interval * (b.iloc[k]['transmitter_x_velocity'] +  x_95 [1] * time_interval * 0.1)
            x_pre_99_low = b.iloc[k]['transmitter_x_position'] + time_interval * (b.iloc[k]['transmitter_x_velocity'] +  x_99 [0] * time_interval * 0.1)
            x_pre_99_up  = b.iloc[k]['transmitter_x_position'] + time_interval * (b.iloc[k]['transmitter_x_velocity'] +  x_99 [1] * time_interval * 0.1)
            y_pre_95_low = b.iloc[k]['transmitter_y_position'] + time_interval * (b.iloc[k]['transmitter_y_velocity'] +  y_95 [0] * time_interval * 0.1)
            y_pre_95_up  = b.iloc[k]['transmitter_y_position'] + time_interval * (b.iloc[k]['transmitter_y_velocity'] +  y_95 [1] * time_interval * 0.1)
            y_pre_99_low = b.iloc[k]['transmitter_y_position'] + time_interval * (b.iloc[k]['transmitter_y_velocity'] +  y_99 [0] * time_interval * 0.1)
            y_pre_99_up  = b.iloc[k]['transmitter_y_position'] + time_interval * (b.iloc[k]['transmitter_y_velocity'] +  y_95 [1] * time_interval * 0.1)
            t_x = 0
            t_y = 0            
            #print(b.iloc[k+1]['tr_x'])
            if b.iloc[k+1]['transmitter_x_position']<=x_pre_95_low or b.iloc[k+1]['transmitter_x_position'] >= x_pre_95_up:
                t_x = 1
            if b.iloc[k+1]['transmitter_x_position']<=x_pre_99_low or b.iloc[k+1]['transmitter_x_position'] >= x_pre_99_up:
                t_x = 2
            if b.iloc[k+1]['transmitter_y_position']<=y_pre_95_low or b.iloc[k+1]['transmitter_y_position'] >= y_pre_95_up:
                t_y = 1
            if b.iloc[k+1]['transmitter_y_position']<=y_pre_99_low or b.iloc[k+1]['transmitter_y_position'] >= y_pre_99_up:
                t_y = 2                 
            score.append(t_x+t_y)
        return np.mean(score)

    def movement_plausibility(self,b):
        score  = 0
        length = b.shape[0] 
        if length <=1:
            score = np.random.randint(2) # why?
            return 0 
        else:
            x_placement        = b.iloc[-1]['transmitter_x_position'] - b.iloc[0]['transmitter_x_position']
            y_placement        = b.iloc[-1]['transmitter_y_position'] - b.iloc[0]['transmitter_y_position']
            average_velocity_x = np.average(b['transmitter_x_velocity'].values)
            average_velocity_y = np.average(b['transmitter_y_velocity'].values)
            if(x_placement==0 and y_placement==0):
                if (average_velocity_x!=0 or average_velocity_y!=0):
                    score = 1.
        return score

    def quantititative_information(self,b):
        length   = b.shape[0] 
        if length == 1: #if there is a single transmission from the emitter
            feature3 = feature4 = feature5 = feature6 = [0]
            return feature3, feature4, feature5, feature6


        time_interval = b.iloc[-1]['reception_time'] - b.iloc[0]['reception_time']
        if time_interval == 0:
            print(b.iloc[-1])
            print(b.iloc[0])
        v_bar_dist_x  = (b.iloc[-1]['transmitter_x_position'] - b.iloc[0]['transmitter_x_position'])/(time_interval)
        v_bar_dist_y  = (b.iloc[-1]['transmitter_y_position'] - b.iloc[0]['transmitter_y_position'])/(time_interval)
            
        # Calculate the v_velocity (v?)
        v_bar_velocity_all_x = 0
        v_bar_velocity_all_y = 0

        for i in range(length-1):
            delta_t = b.iloc[i+1]['reception_time'] - b.iloc[i]['reception_time']
            v_bar_velocity_all_x = v_bar_velocity_all_x + b.iloc[i]['transmitter_x_velocity'] * delta_t
            v_bar_velocity_all_y = v_bar_velocity_all_y + b.iloc[i]['transmitter_y_velocity'] * delta_t
        
        v_bar_velo_x = v_bar_velocity_all_x/time_interval
        v_bar_velo_y = v_bar_velocity_all_y/time_interval 
        feature3 = np.abs(v_bar_dist_x - v_bar_velo_x) #v_measure_x
        feature4 = np.abs(v_bar_dist_y - v_bar_velo_y) #v_measure_y
        feature5 = np.linalg.norm([feature3 , feature4]) # v_mag 
        #v_total 
        feature6 = np.abs(np.linalg.norm([v_bar_dist_x*time_interval, v_bar_dist_y*time_interval]) - np.linalg.norm([v_bar_velocity_all_x, v_bar_velocity_all_y]))
        return feature3, feature4, feature5, feature6

    
# vecotrized versions (10x faster but still slow)
    
    def location_plausibility_score(self,df,next_df,TI):
        x_95 = [- 5.6983,  5.2265] 
        x_99 = [- 7.1795,  7.7077]
        y_95 = [- 8.1203,  8.0501]
        y_99 = [- 12.1629, 12.0927]

        def sub_score(series_position,series_velocity,TI,acc,series_next_position):
            return  ((series_next_position<= (series_position+TI*(series_velocity+acc[0]*0.1*TI))) | (series_next_position>=(series_position+TI*(series_velocity+acc[1]*0.1*TI)) )).astype(int)

        t_x95=sub_score(df.transmitter_x_position,df.transmitter_x_velocity,TI, x_95,next_df.transmitter_x_position)*df.transmitter_x_velocity.clip(upper=1,lower=1)
        t_x99=sub_score(df.transmitter_x_position,df.transmitter_x_velocity,TI, x_99,next_df.transmitter_x_position)*df.transmitter_x_velocity.clip(upper=1,lower=1)
        t_y95=sub_score(df.transmitter_y_position,df.transmitter_y_velocity,TI, y_95,next_df.transmitter_y_position)*df.transmitter_x_velocity.clip(upper=1,lower=1)
        t_y99=sub_score(df.transmitter_y_position,df.transmitter_y_velocity,TI, y_99,next_df.transmitter_y_position)*df.transmitter_x_velocity.clip(upper=1,lower=1)
        score = (t_x95+t_x99)+(t_y95+t_y99)
        return df.assign(score=score.fillna(2)).groupby(['transmitter_id','receiver_id'])['score'].mean()


    def movement_plausibility_vectorised(self,df):
        return df.groupby(['transmitter_id','receiver_id'],as_index=True).apply(lambda x :
                                                                            0 if len(x)<=1 else 
                                                                            int(((x.iloc[-1]['transmitter_x_position'] - x.iloc[0]['transmitter_x_position'])==0) &
                                                                            ((x.iloc[-1]['transmitter_y_position'] - x.iloc[0]['transmitter_y_position'])==0) &
                                                                            ((x['transmitter_x_velocity'].mean()!=0)  | (x['transmitter_y_velocity'].mean()!=0)))

                                                                           )
    
    
    def add_feature_vectors_vectorized(self):
        sorted_df=self.df.sort_values(['transmitter_id','receiver_id','reception_time'])

        shifted_df=sorted_df.groupby(['transmitter_id','receiver_id'],as_index=True).apply(lambda x :x.shift())

        shifted_df.transmitter_id = sorted_df.transmitter_id
        shifted_df.receiver_id = sorted_df.receiver_id

        TI = sorted_df.reception_time-shifted_df.reception_time
        feature1 = self.location_plausibility_score(shifted_df,sorted_df,TI)
        feature2 = self.movement_plausibility_vectorised(sorted_df)

        feature3 = sorted_df.assign(local_velocity=TI*shifted_df.transmitter_x_velocity).groupby(['transmitter_id','receiver_id'],as_index=True).apply(lambda x :
                                                                                0 if (x.iloc[-1]['reception_time'] - x.iloc[0]['reception_time'])==0 else
                                                                                abs((x.iloc[-1]['transmitter_x_position'] - x.iloc[0]['transmitter_x_position']) - x.local_velocity.sum())/(x.iloc[-1]['reception_time'] - x.iloc[0]['reception_time'])
                                                                               )

        feature4 = sorted_df.assign(local_velocity=TI*shifted_df.transmitter_y_velocity).groupby(['transmitter_id','receiver_id'],as_index=True).apply(lambda x :
                                                                                0 if (x.iloc[-1]['reception_time'] - x.iloc[0]['reception_time'])==0 else
                                                                                abs((x.iloc[-1]['transmitter_y_position'] - x.iloc[0]['transmitter_y_position']) - x.local_velocity.sum())/(x.iloc[-1]['reception_time'] - x.iloc[0]['reception_time'])
                                                                               )
        feature5 = np.sqrt(feature3**2+feature4**2)



        feature6=sorted_df.assign(local_velocity_y=TI*shifted_df.transmitter_y_velocity,local_velocity_x=TI*shifted_df.transmitter_x_velocity).groupby(['transmitter_id','receiver_id'],as_index=True).apply(lambda x :
                                                                                0 if (x.iloc[-1]['reception_time'] - x.iloc[0]['reception_time'])==0 else
                                                                                abs(np.linalg.norm([x.iloc[-1]['transmitter_x_position'] - x.iloc[0]['transmitter_x_position'],
                                                                                                    x.iloc[-1]['transmitter_y_position'] - x.iloc[0]['transmitter_y_position']]) 

                                                                                    - 

                                                                                    np.linalg.norm([x.local_velocity_x.sum()
                                                                                                  ,
                                                                                                   x.local_velocity_y.sum()]
                                                                                                  )

                                                                               ))

        catdf=pd.concat([feature1,feature2,feature3,feature4,feature5,feature6],axis=1)
        catdf.columns=['feature1','feature2','feature3','feature4','feature5','feature6']
        return sorted_df.groupby(['transmitter_id','receiver_id']).head(1).merge(catdf.reset_index(),on=['transmitter_id','receiver_id'])

    #def distancebis(self):
        #self.df = self.df.assign(distance2 = np.sqrt((self.df.transmitter_x_position-self.df.receiver_x_position)**2 + (self.df.transmitter_y_position-self.df.receiver_y_position)**2))
        #Faux#
        #self.df = self.df.assign(distance3 = np.linalg.norm([self.df.receiver_x_position-self.df.transmitter_x_position, self.df.receiver_y_position-self.df.transmitter_y_position]))
    
    # def check_range(data, threshold): # We obtain the index for which the distance between sender and receiver is higher 800.
    #     length     = data.shape[0]
    #     drop_index = np.zeros(length)
    #     for i in range(length):
    #         distance_this = distance(data.iloc[i])
    #         if distance_this > threshold:
    #             drop_index[i]=1
    #     drop_index_ = np.where(drop_index>0)
    #     drop_row    = np.asarray(drop_index_)[0]
    #     return drop_row

## Feature Vectors

    # def location_plausibility(receiver_of_sender):
    #    #x_95 = [-5.6983, 5.2265]
    #    #x_99 = [-7.1795, 7.7077]
    #    #y_95 = [-8.1203, 8.0501]
    #    #y_99 = [-12.1629, 12.0927]   

    #     x_95   = [-10, 10] # What is the difference?
    #     x_99   = [-18, 18] # What is the difference?
    #     y_95   = [-10, 10] # What is the difference?
    #     y_99   = [-18, 18] # What is the difference?   
    #     score  = []
    #     length = receiver_of_sender.shape[0]
    #     score.append(2) # for the start of the series, we think it is two. 
    #     if length <=1:
    #         return score
    #     for k in range(length-1):
    #             time_interval = (receiver_of_sender.iloc[k+1]['re_time'] - receiver_of_sender.iloc[k]['re_time'])
    #             x_pre_95_low = receiver_of_sender.iloc[k]['tr_x'] + time_interval * (receiver_of_sender.iloc[k]['tr_vx'] +  x_95 [0] * time_interval * 0.1)
    #             x_pre_95_up  = receiver_of_sender.iloc[k]['tr_x'] + time_interval * (receiver_of_sender.iloc[k]['tr_vx'] +  x_95 [1] * time_interval * 0.1)
    #             x_pre_99_low = receiver_of_sender.iloc[k]['tr_x'] + time_interval * (receiver_of_sender.iloc[k]['tr_vx'] +  x_99 [0] * time_interval * 0.1)
    #             x_pre_99_up  = receiver_of_sender.iloc[k]['tr_x'] + time_interval * (receiver_of_sender.iloc[k]['tr_vx'] +  x_99 [1] * time_interval * 0.1)
    #             y_pre_95_low = receiver_of_sender.iloc[k]['tr_y'] + time_interval * (receiver_of_sender.iloc[k]['tr_vy'] +  y_95 [0] * time_interval * 0.1)
    #             y_pre_95_up  = receiver_of_sender.iloc[k]['tr_y'] + time_interval * (receiver_of_sender.iloc[k]['tr_vy'] +  y_95 [1] * time_interval * 0.1)
    #             y_pre_99_low = receiver_of_sender.iloc[k]['tr_y'] + time_interval * (receiver_of_sender.iloc[k]['tr_vy'] +  y_99 [0] * time_interval * 0.1)
    #             y_pre_99_up  = receiver_of_sender.iloc[k]['tr_y'] + time_interval * (receiver_of_sender.iloc[k]['tr_vy'] +  y_95 [1] * time_interval * 0.1)
    #             t_x = 0
    #             t_y = 0            
    #             print(receiver_of_sender.iloc[k+1]['tr_x'])
    #             if receiver_of_sender.iloc[k+1]['tr_x']<=x_pre_95_low or receiver_of_sender.iloc[k+1]['tr_x'] >= x_pre_95_up:
    #                 t_x = 1
    #             if receiver_of_sender.iloc[k+1]['tr_x']<=x_pre_99_low or receiver_of_sender.iloc[k+1]['tr_x'] >= x_pre_99_up:
    #                 t_x = 2
    #             if receiver_of_sender.iloc[k+1]['tr_y']<=y_pre_95_low or receiver_of_sender.iloc[k+1]['tr_y'] >= y_pre_95_up:
    #                 t_y = 1
    #             if receiver_of_sender.iloc[k+1]['tr_y']<=y_pre_99_low or receiver_of_sender.iloc[k+1]['tr_y'] >= y_pre_99_up:
    #                 t_y = 2                 
    #             score.append(t_x+t_y)
    #     return np.mean(score)

    # def movement_plausibility(receiver_of_sender):
    #     flag  = 0.
    #     length = receiver_of_sender.shape[0]
    #     if length <=1:
    #         flag = np.random.randint(2)
    #         return flag 
    #     x_placement        = receiver_of_sender.iloc[-1]['tr_x'] - receiver_of_sender.iloc[0]['tr_x']
    #     y_placement        = receiver_of_sender.iloc[-1]['tr_y'] - receiver_of_sender.iloc[0]['tr_y']
    #     average_velocity_x = np.average(receiver_of_sender['tr_vx'].values)
    #     average_velocity_y = np.average(receiver_of_sender['tr_vy'].values)
    #     if(x_placement==0 and y_placement==0):
    #         if (average_velocity_x!=0 or average_velocity_y!=0):
    #             flag = 1.
    #     return flag

    # def quantititative_information(receiver_of_sender):
    #     feature3 = []
    #     feature4 = []
    #     feature5 = []
    #     feature6 = [] 
    #     length = receiver_of_sender.shape[0] 
        
    #     if length == 1:
    #         feature3 = [0]
    #         feature4 = [0]
    #         feature5 = [0]
    #         feature6 = [0]
    #         return feature3, feature4, feature5, feature6
    #     time_interval = receiver_of_sender.iloc[-1]['re_time'] - receiver_of_sender.iloc[0]['re_time']
    #     v_bar_dist_x = (receiver_of_sender.iloc[-1]['tr_x'] - receiver_of_sender.iloc[0]['tr_x'])/(time_interval)
    #     v_bar_dist_y = (receiver_of_sender.iloc[-1]['tr_y'] - receiver_of_sender.iloc[0]['tr_y'])/(time_interval)
        
    #     # Calculate the v_velocity
    #     v_bar_velocity_all_x = 0
    #     v_bar_velocity_all_y = 0
    #     for i in range(length-1):
    #         delta_t = receiver_of_sender.iloc[i+1]['re_time'] - receiver_of_sender.iloc[i]['re_time']
    #         v_bar_velocity_all_x = v_bar_velocity_all_x + receiver_of_sender.iloc[i]['tr_vx'] * delta_t
    #         v_bar_velocity_all_y = v_bar_velocity_all_y + receiver_of_sender.iloc[i]['tr_vy'] * delta_t
    #     v_bar_velo_x = v_bar_velocity_all_x/time_interval
    #     v_bar_velo_y = v_bar_velocity_all_y/time_interval 
    #     v_measure_x = np.abs(v_bar_dist_x - v_bar_velo_x)
    #     v_measure_y = np.abs(v_bar_dist_y - v_bar_velo_y)
    #     v_mag = np.linalg.norm([v_measure_x, v_measure_y])
    #     v_total  = np.abs(np.linalg.norm([v_bar_dist_x*time_interval, v_bar_dist_y*time_interval]) - np.linalg.norm([v_bar_velocity_all_x, v_bar_velocity_all_y]))
    #     # remove the extra lines
    #     feature3 = v_measure_x
    #     feature4 = v_measure_y
    #     feature5 = v_mag 
    #     feature6 = v_total
    #     return feature3, feature4, feature5, feature6

    # def distance_check(receiver_of_sender, threshold):
    #     length = receiver_of_sender.shape[0]
    #     distance_score = []
    #     for i in range(length):
    #         distance_score.append(0)
    #         x = receiver_of_sender.iloc[i]['tr_x'] - receiver_of_sender.iloc[i]['re_x']
    #         y = receiver_of_sender.iloc[i]['tr_y'] - receiver_of_sender.iloc[i]['re_y']
    #         distance = np.linalg.norm([x,y])
    #         if distance >= 800:
    #             distance_score[i] = 1
    #     return np.mean(distance_score)
