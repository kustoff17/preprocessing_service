import numpy as np
import pandas as pd
import seaborn as sns

class Preprocessor:
    def __init__(self,df):
        self.df = df
        self.num_features = [col for col in self.df.columns if df[col].dtype!='object']
        self.cat_features = [col for col in self.df.columns if col not in self.num_features]

    # среднне по каждой числвоой фиче
    def mean(self):
        return self.df[self.num_features].mean()


    #заполнение пропусков
    def  fill_nan_values(self):
        for col in self.df.columns:
            if col in self.num_features:
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                self.df[col] = self.df[col].fillna(self.df[col].mode().iloc[0])



    # encoding сделать и учесть условие не более 25 разл значений

    #удаление высокоскореллированных фичей
    def del_corr_features(self, trashhold = 0.5):
        corr_matrix = self.df[self.num_features].corr().abs()
        corr_matrix_triu = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) 
        to_del = []
        for col in corr_matrix_triu.columns:
            if ((corr_matrix_triu[col] >= trashhold) &(corr_matrix_triu[col] != 1)):
                to_del.append(col)
        self.df.drop(columns = to_del, inplace=True)

    # энкодинг кат фичей, + числовые с 25 уник  знач
    def encoding_cat_fetures(self, target = None):
        for col in (self.cat_features+[num for num in self.num_features if self.df[num].nunique() >=25]):
            if len(self.df[col].unique()) <= 5:
                self.df = pd.get_dummies(self.df, columns=[col])
            else:
                if target is not None and target in self.df.columns:
                    encode_col = self.df.groupby(col)[target].mean()
                    self.df[col] = self.df[col].apply(lambda x: encode_col.loc[x])

                # если знаечение для таргета не нашлось, применяем label
                else:
                    mapping  ={val: index for index, val in enumerate(self.df[col].unique())}
                    self.df[col] = self.df[col].map(mapping)



    def discret_ddm(self, col = 'create_dttm'):
        if col in self.df.columns:
            self.df[col] = pd.to_datetime(self.df[col])
            self.df['Year']=self.df[col].dt.year
            self.df['Month'] = self.df[col].dt.month
            self.df['Day'] = self.df[col].dt.day



















