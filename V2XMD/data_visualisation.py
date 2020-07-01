import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd

sns.set_style('dark')

attack_types ={
    0: 'Genuine',
    1: 'Constant',
    2: 'Constant offset',
    4: 'Random',
    8: 'Random offset',
    16:'Eventual stop'
}

attack_color_maps={
    0 :plt.cm.Greys,
    1 :plt.cm.Purples,
    2 :plt.cm.Blues,
    4 :plt.cm.Greens,
    8 :plt.cm.Reds,
    16:plt.cm.Oranges,
}

attack_to_int={
 0 : 0,
 1 : 1,
 2 : 2,
 4 : 3,
 8 : 4,
 16: 5
 }

class DataVisualizer:
    def __init__(self,df):
        self.df =df
        self.fontdict={'fontsize': 16,
                       'weight' : 'bold',
                       'horizontalalignment': 'center'
                       }
        self.pca_grps=None
        
    def transmissionRatePiePlot(self):
        transmission_counts=self.df.groupby(['dataset','receiver_id','transmitter_id']).size().value_counts().sort_index()
        supp15 = transmission_counts[15:].sum()
        transmission_counts=transmission_counts[:14]
        transmission_counts['15+']= supp15
        transmission_counts.plot.pie(autopct='%1.0f%%',pctdistance=0.8, labeldistance=1.1,radius=1,cmap = plt.cm.coolwarm,label="", fontsize=15,figsize=(10,10),explode=[0.01]*14+[0.05])
        plt.title("Number of messages exchanged between unique pairs of vehicles",fontdict=self.fontdict)

    def attackProps(self):
        self.df.label=self.df.label.astype(int)
        props=self.df.groupby(['dataset','label']).size()
        [(plt.subplot(2,3,i+1),
          props[x].plot.pie(autopct='%1.0f%%',pctdistance=0.6, labeldistance=1.1,radius=1.25,cmap = plt.cm.coolwarm,label="", fontsize=15)
         ,plt.title(attack_types[int(x[2:])])
         ) for i, x in enumerate(props.index.get_level_values(0).unique())]
        plt.suptitle("Attack proportions per dataset",fontdict=self.fontdict)

    def nbMessagesDistribution(self):
        plt.figure(dpi=300)
        ax = plt.subplot(3,1,2)
        ax.boxplot(self.df[['transmitter_id','bsm_id']].drop_duplicates().groupby('transmitter_id').size(),vert=False)
        plt.title('number of message per vehicle')
        plt.tight_layout()
        
    def distanceDistribution(self):
        plt.figure(dpi=300)
        ax = plt.subplot(3,1,2)
        ax.boxplot(self.df.distance,vert=False,showmeans=True,showfliers=True)
        ax = plt.subplot(3,1,1)
        ax.boxplot(self.df.distance,vert=False,showmeans=True,showfliers=False)
        plt.title('distance')
        plt.tight_layout()
    
    def speedDistribution(self):
        plt.figure(dpi=300)
        ax = plt.subplot(3,1,2)
        ax.boxplot(self.df.speed,vert=False,showmeans=True)
        plt.title('speed')
        plt.tight_layout()
        
    def distanceDistributionPerDataSet(self):
        self.df.boxplot(by='dataset',column='distance',vert=False,showmeans=True,figsize=(16,12))
        plt.suptitle('')
        plt.xlabel('distance (m)',fontdict=self.fontdict)
        plt.ylabel('dataset',fontdict=self.fontdict)
        plt.title("distance per dataset")
    def printStats(self):
        stats_series=pd.Series({
        'N° messages':self.df.bsm_id.size,
        'N° unique messages':self.df.bsm_id.unique().size,
        'N° vehicles':self.df.transmitter_id.unique().size,
        'Avreage RSSI': self.df.rssi.mean(),
        })
        print(stats_series)

    def statsPlot(self):
        self.nbMessagesDistribution()          # nb_messages distribution
        self.distanceDistribution()            # distance distribution
        self.speedDistribution()               # speed ditribution
        self.distanceDistributionPerDataSet()  # distance distribution per data set
        self.printStats()                      # some quantitaives states
        
    def plotFeaturesPairPlots(self,feature1='receiver_x_position',feature2='transmitter_x_position',share_axes=False, plot_genuins=False,figsize=(15,9)):
        if share_axes : 
            ax = plt.subplots(2,3,figsize=figsize,sharex=True,sharey=True)
        else :
            ax = plt.subplots(2,3,figsize=figsize)
        grouped_data = self.df.groupby('label')[[feature1,feature2]]
        for name,grp in grouped_data:
            if share_axes:
                plt.subplot(2,3,attack_to_int[name]+1,sharex=ax[0].axes[0],sharey=ax[0].axes[0])
            else :
                plt.subplot(2,3,attack_to_int[name]+1)


            plt.title(attack_types[name],fontdict=self.fontdict)
            plt.hexbin(*grp[[feature1,feature2]].T.values ,bins='log',cmap=plt.cm.hot,alpha=0.75)
            if plot_genuins:
                plt.hexbin(*grouped_data.get_group(0)[[feature1,feature2]].T.values ,bins='log',cmap=attack_color_maps[0],alpha=0.4,mincnt=50)

            plt.xlabel(feature1)
            plt.ylabel(feature2)
            ax = plt.gca()
            ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c="blue",lw=2,label='true data tendency')
            ax.set_facecolor('black')
            ax.legend(frameon=True)

        plt.tight_layout()
        
        
    def runPCA(self,features=[]):
        if len(features)==0:
            features=self.df.columns[~self.df.columns.isin(self.df.dtypes[self.df.dtypes=='object'].index.tolist()+['label'])]
        pca_df=self.df[features]
        pca=sklearn.decomposition.PCA(2)
        pca_fit=pca.fit(pca_df)
        self.pca_grps = pca_df.groupby(self.df.label).apply(lambda x : pd.DataFrame(pca_fit.transform(x)))
        
    def __check_pca(self):
        if self.pca_grps is None:
            print("please run pca first")
            return False
        return True
    
    def pca_all_1d_kde(self,axx):
        if not self.__check_pca():
            return None
        plt.figure(figsize=(16,16))
        for x in set(self.pca_grps.index.get_level_values(0)) :
            sns.kdeplot(self.pca_grps[axx][x].values, alpha=0.4,label=attack_types[x])
        plt.legend([attack_types[x] for x in [0,1,2,4,8,16]])

    def pca_all_hist(self,axx):
        if not self.__check_pca():
            return None
        plt.figure(figsize=(16,16))
        for x in set(self.pca_grps.index.get_level_values(0)) :

            plt.hist(self.pca_grps[axx][x].values, 100, density=False, histtype='stepfilled', alpha=0.4)
            plt.legend([attack_types[x] for x in [0,1,2,4,8,16]])

    def pca_all_hexbin(self):
        if not self.__check_pca():
            return None
        
        ax=plt.subplots(2,3,figsize=(18,18),sharex=True,sharey=True)
        [(plt.subplot(2,3,attack_to_int[x]+1),
          plt.hexbin(self.pca_grps.loc[x].values.T[0],self.pca_grps.loc[x].values.T[1],cmap=attack_color_maps[x],bins='log',mincnt=25,label=attack_types[x]),
          plt.title(attack_types[x])
         ) for x in set(self.pca_grps.index.get_level_values(0))]

    def pca_all_2d_kde(self):
        if not self.__check_pca():
            return None
        pass
        plt.figure(figsize=(16,16))
        for x in set(self.pca_grps.index.get_level_values(0)) :
            sns.kdeplot(self.pca_grps.loc[x][0],self.pca_grps.loc[x][1],cmap=attack_color_maps[x],label=attack_types[x])
        plt.legend([attack_types[x] for x in [0,1,2,4,8,16]])