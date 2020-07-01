import pandas as pd
import sklearn


class Evaluation:
    def __init__(self,df):
        self.df=df
    def precisionRecallByDataSet(self):
        return self.df.groupby(['dataset'])['truth','pred'].apply(
                        lambda x : pd.Series(
                                        {
                                            'precision':sklearn.metrics.precision_score(x['truth'],x['pred']),
                                            'recall'   :sklearn.metrics.recall_score   (x['truth'],x['pred'])
                                        }) 
                        )
    def confusionMatrixByDataSet(self):
        return self.df.groupby(['dataset']).apply(
                        lambda x : pd.DataFrame(
                                                    sklearn.metrics.confusion_matrix(x['truth'],x['pred'],labels=[True,False]),columns=[True,False],index=[True,False])
                                               )