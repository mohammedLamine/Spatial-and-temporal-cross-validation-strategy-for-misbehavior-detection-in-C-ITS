from matplotlib.colors import ListedColormap
import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt

import V2XMD

irrelevant_train_columns=[  'label',
                            "type",
                            "receiver_id",
                            "receiver_z_position",
                            "transmission_time" ,
                            "transmitter_id",
                            "bsm_id",
                            "transmitter_z_position",
                            "transmitter_z_velocity"]

train_columns = [
                "receiver_x_position",
                "receiver_y_position",
                "transmitter_x_position",
                "transmitter_y_position",
                "transmitter_x_velocity",
                "transmitter_y_velocity",
                "rssi", 'speed',
       'distance', 'ssc', 'saw', 'art'   
]

class TrainModel:
    def __init__(self,df,split='random',multi_class=False,features=False,nb_leaks=100):
        """
        Contsructor for the model training class
        df is the dataframe containing the messages
        split is the type of split to use (random, our split)
        multi_class is a parameter of whether to build a model for each class of attack or one model for all
        features decides whether to use extracted features alongside the message features
        """
        self.df =df
        self.split=split
        self.nb_leaks=nb_leaks
        self.multi_class=multi_class
        self.features=features
        self.names = ["Nearest Neighbors", 
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes"]
        if features :
            self.addMultipleFeaturesToData()
        """
        the set of tested classifiers
        """
        self.classifiers = [
            KNeighborsClassifier(5,n_jobs=10),
            DecisionTreeClassifier(),
            RandomForestClassifier(n_estimators=10,n_jobs=10),
            MLPClassifier(max_iter=700),
            AdaBoostClassifier(),
            GaussianNB()]
        self.train_results=None
        self.test_results=None
        
    def join_groupby(df):
        df=pd.concat([grp for idgrp,grp in  df])
        if type(df) is pd.DataFrame :
            sub_train_columns= list(df.columns.intersection(train_columns))
            return df[sub_train_columns]
        return df

    def train(self):
        """
        function used to split and train the model
        """
        if self.split.lower() == 'random':
            print('random splitting')
            X_train, y_train,X_valid, y_valid,X_test, y_test=TrainModel.random_train_split(self.df)
        elif self.split.lower()== 'our' :
            print('our')
            X_train, y_train,X_valid, y_valid,X_test, y_test=TrainModel.our_train_split(self.df)
        elif self.split.lower()== 'leakage' :
            X_train, y_train,X_valid, y_valid,X_test, y_test=TrainModel.leakage_split(self.df,nb_leaks=self.nb_leaks)
        rsd_dict={}
        test_dict={}
        if not self.multi_class :
            for grp_name in X_train.indices.keys():
                results={}
                results_test={}

                for name, clf in zip(self.names,self.classifiers):
                    clf.fit(X_train.get_group(grp_name), y_train.get_group(grp_name))
                    results[name]=sklearn.metrics.classification_report( y_valid.get_group(grp_name),clf.predict(X_valid.get_group(grp_name)),output_dict=True,zero_division=0)
                    results_test[name]=sklearn.metrics.classification_report( y_test.get_group(grp_name),clf.predict(X_test.get_group(grp_name)),output_dict=True,zero_division=0)

                rsd_dict[grp_name]=results
                test_dict[grp_name]=results_test

        else :
            results={}
            results_test={}

            for name, clf in zip(self.names, self.classifiers):
                clf.fit(TrainModel.join_groupby(X_train), TrainModel.join_groupby(y_train))
                results[name]=sklearn.metrics.classification_report( TrainModel.join_groupby(y_valid),clf.predict(TrainModel.join_groupby(X_valid)),output_dict=True,zero_division=0)
                results_test[name]=sklearn.metrics.classification_report( TrainModel.join_groupby(y_test),clf.predict(TrainModel.join_groupby(X_test)),output_dict=True,zero_division=0)
            rsd_dict['grouped']=results
            test_dict['grouped']=results_test
            
            
        self.train_results=self.results_to_df(rsd_dict,w_set='train')
        self.test_results=self.results_to_df(test_dict,w_set='test')


    def results_to_df(self,dct,w_set):
        """
        Making a dataframe of all obtained results that is easy to use for visualisation
        """
        rsdf=pd.DataFrame(dct).stack()
        rsdfsub=rsdf.apply(lambda x: pd.Series(x)).stack().apply(lambda x: pd.Series(x))
        rsdfsub=rsdfsub.droplevel(1).reset_index().assign(level_2='grouped').groupby(['level_0','level_2','level_1']).mean()
        rsdfsub=rsdfsub.loc[self.names,:,list(V2XMD.data_visualisation.attack_types.values())][['precision','recall','f1-score']].droplevel(1).unstack(0)
        rsdfsub=rsdfsub.swaplevel(axis=1).sort_index(axis=1).assign(
            split=self.split,
            Set = w_set,
            multiclass=str(self.multi_class),
            features=str(self.features)
        ).set_index(['split','Set','features','multiclass'],append=True)
        rsdfsub = rsdfsub.rename_axis(columns={'level_0':'Algo'},index={'level_1':'Attack'})
        return rsdfsub
    
    def results_visualisation(self,split='test'):
        """
        Histograms and boxplots of the obtained performance
        """
        if split == 'test':
            precisionrecall_df=self.test_results.swaplevel(axis=1)
        elif split == 'train':
            precisionrecall_df=self.train_results.swaplevel(axis=1)
        else : 
            print('split is wrong')
            return None
        precisionrecall_df['recall'].plot.bar(figsize=(7,2.3),legend=None,rot=35)
        plt.legend(ncol=4,loc=9,bbox_to_anchor=(0.5, 1.3))
        ticklabels=self.train_results.index.get_level_values('Attack').unique().values
        plt.xticks(ticks=range(len(ticklabels)),labels=ticklabels)
        plt.ylabel('recall')
        plt.xlabel('Attack')
        plt.savefig("out/recall_no_features.png",dpi=300,bbox_inches='tight')
        precisionrecall_df['precision'].plot.bar(figsize=(7,2.3),legend=None,rot=35)
        plt.legend(ncol=4,loc=9,bbox_to_anchor=(0.5, 1.3))
        plt.xticks(ticks=range(len(ticklabels)),labels=ticklabels)
        plt.ylabel('precision')
        plt.xlabel('Attack')
        plt.savefig("out/precision_no_features.png",dpi=300,bbox_inches='tight')
        

    def addMultipleFeaturesToData(self):
        """
        Extract features (ssc, saw, art) and add it to the dataframe for training 
        """
        
        dprocess = V2XMD.data_processing.DataProcessing(self.df.sort_values('reception_time').copy())
        dprocess.add_speed()
        dprocess.add_distance()
        pc        = V2XMD.plausibility_checks.PlausibilityChecks(dprocess.df)
        sscres=pc.ssc(10)
        sawres=pc.saw(200)
        artres=pc.art(500)

        featuredf=dprocess.df.copy()
        featuredf=featuredf.assign(ssc=sscres.pred)
        featuredf=featuredf.assign(saw=sawres.pred)
        featuredf=featuredf.assign(art=artres.pred)

        self.df = featuredf
    
    
    def random_train_split(df,valid_frac=0.8,test_frac=0.8):
        """
        random split using sklearn train_test split function
        """
        sub_train_columns= list(df.columns.intersection(train_columns))
        train_data, test_data = df.sort_values('reception_time')[:int(df.shape[0]*(test_frac))], df.sort_values('reception_time')[int(df.shape[0]*(test_frac)):]
        train_data, valid_data = sklearn.model_selection.train_test_split(train_data,test_size=1-valid_frac)

        X_train, y_train= train_data.drop(irrelevant_train_columns,axis=1).groupby('dataset')[sub_train_columns], train_data.groupby('dataset').label
        X_test, y_test =test_data.drop(irrelevant_train_columns,axis=1).groupby('dataset')[sub_train_columns], test_data.groupby('dataset').label
        X_valid, y_valid =valid_data.drop(irrelevant_train_columns,axis=1).groupby('dataset')[sub_train_columns], valid_data.groupby('dataset').label

        return X_train, y_train,X_valid, y_valid,X_test, y_test

    def our_train_split(df,valid_frac=0.8,test_frac=0.8):
        """
        our proposed split with our  and spatial considerations
        """
        sub_train_columns= list(df.columns.intersection(train_columns))
        train_data, test_data = df.sort_values('reception_time')[:int(df.shape[0]*(test_frac))], df.sort_values('reception_time')[int(df.shape[0]*(test_frac)):]
        intersection_idxs=test_data.bsm_id.isin(np.intersect1d(train_data.bsm_id,test_data.bsm_id))
        train_data= pd.concat([train_data, test_data[intersection_idxs]])
        test_data = test_data[~intersection_idxs]
        
        train_data, valid_data = train_data.sort_values('reception_time')[:int(train_data.shape[0]*(valid_frac))], train_data.sort_values('reception_time')[int(train_data.shape[0]*(valid_frac)):]
        
        intersection_idxs=valid_data.bsm_id.isin(np.intersect1d(train_data.bsm_id,valid_data.bsm_id))
        train_data = pd.concat([train_data, valid_data[intersection_idxs]])
        train_data = train_data.sort_values('reception_time')
        valid_data = valid_data[~intersection_idxs]
        
        X_train, y_train= train_data.drop(irrelevant_train_columns,axis=1).groupby('dataset')[sub_train_columns], train_data.groupby('dataset').label
        X_test, y_test =test_data.drop(irrelevant_train_columns,axis=1).groupby('dataset')[sub_train_columns], test_data.groupby('dataset').label
        X_valid, y_valid =valid_data.drop(irrelevant_train_columns,axis=1).groupby('dataset')[sub_train_columns], valid_data.groupby('dataset').label

        return X_train, y_train,X_valid, y_valid,X_test, y_test
    
    
    def leakage_split(df,valid_frac=0.8,test_frac=0.8,nb_leaks=100):
        """
        Intentional leakage split for investigaion of the impact of leakage on the performance
        """
        sub_train_columns= list(df.columns.intersection(train_columns))

        train_data, test_data = df.sort_values('reception_time')[:int(df.shape[0]*(test_frac))], df.sort_values('reception_time')[int(df.shape[0]*(test_frac)):]

        intersection_idxs=test_data.bsm_id.isin(np.intersect1d(train_data.bsm_id,test_data.bsm_id))

        train_data= pd.concat([train_data, test_data[intersection_idxs]])

        test_data = test_data[~intersection_idxs]

        train_data, valid_data = train_data.sort_values('reception_time')[:int(train_data.shape[0]*(valid_frac))], train_data.sort_values('reception_time')[int(train_data.shape[0]*(valid_frac)):]
        intersection_idxs=valid_data.bsm_id.isin(np.intersect1d(train_data.bsm_id,valid_data.bsm_id))
        train_data = pd.concat([train_data, valid_data[intersection_idxs]])
        train_data = train_data.sort_values('reception_time')
        valid_data = valid_data[~intersection_idxs]
        
        # leakage application
        leakable_train=train_data.bsm_id.value_counts()[train_data.bsm_id.value_counts()>=2].index
        print("leakable train messages :", len(leakable_train))

        if len(leakable_train)<nb_leaks: 
            print("number of leaks excess the number of unique messages with 2 minimum copys. TLRD: can't leak that many",len(leakable_train),nb_leaks)
        leaking_train_idx=leakable_train[np.random.choice(len(leakable_train),size=nb_leaks,replace=False)]
        train_samples_to_leak = np.concatenate(train_data[train_data.bsm_id.isin(leaking_train_idx)]
                                               .groupby('bsm_id')
                                               .apply(lambda x, size : sklearn.model_selection.train_test_split(x.index,test_size=size)
                                                      [0].values,size=0.5).values
                                              )
        leakable_valid=valid_data.bsm_id.value_counts()[valid_data.bsm_id.value_counts()>=2].index
        print("leakable validaion messages :", len(leakable_valid))
        if len(leakable_valid)<nb_leaks: 
            print("number of leaks excess the number of unique messages with 2 minimum copys. TLRD: can't leak that many",len(leakable_valid),nb_leaks)

        leaking_valid_idx=leakable_valid[np.random.choice(len(leakable_valid),size=nb_leaks,replace=False)]

        valid_samples_to_leak = np.concatenate(valid_data[valid_data.bsm_id.isin(leaking_valid_idx)]
                                           .groupby('bsm_id')
                                           .apply(lambda x, size : sklearn.model_selection.train_test_split(x.index,test_size=size)
                                                  [0].values,size=0.5).values
                                          )

        train_data = pd.concat([train_data, valid_data.loc[valid_samples_to_leak]])
        valid_data = pd.concat([valid_data, train_data.loc[train_samples_to_leak]])

        valid_data = valid_data[~valid_data.index.isin(valid_samples_to_leak)]
        train_data = train_data[~train_data.index.isin(train_samples_to_leak)]

        train_data = train_data.sort_values('reception_time')
        valid_data = valid_data.sort_values('reception_time')

        X_train, y_train= train_data.drop(irrelevant_train_columns,axis=1).groupby('dataset')[sub_train_columns], train_data.groupby('dataset').label
        X_test, y_test =test_data.drop(irrelevant_train_columns,axis=1).groupby('dataset')[sub_train_columns], test_data.groupby('dataset').label
        X_valid, y_valid =valid_data.drop(irrelevant_train_columns,axis=1).groupby('dataset')[sub_train_columns], valid_data.groupby('dataset').label

        return X_train, y_train,X_valid, y_valid,X_test, y_test