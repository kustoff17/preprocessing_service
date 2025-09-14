import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class Preprocessor:
    def __init__(self,df):
        self.df = df
        self.num_features = [col for col in self.df.columns if df[col].dtype!='object']
        self.cat_features = [col for col in self.df.columns if col not in self.num_features]

    # среднне по каждой числвоой фиче
    def mean(self):
        return self.df[self.num_features].mean()

    # исправить вывод табличный (теперь выводится как dataframe)
    def __str__(self):
        return self.df.head(7).to_string()

    def add_hours_interval_feature(self):
         self.df['hours_intervals'] = self.df['Hour'].apply(lambda x : 0 if 0<=x<=8 else 1)
         self.set_features() # обновление кортежа столбцов

    # добавлена фича, точнее определяющая время дня
    def add_part_of_day(self):
        self.df['part_of_day'] = self.df['Hour'].apply(lambda x: 'morning' if 0<= x<=8 else ('day' if 8<x<=15 else 'evening'))
        self.set_features()  #обновление кортежа столбцов


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
            if any(corr_matrix_triu[col].dropna() >= trashhold): #any смотрит наличие во всем столбце
                to_del.append(col)
        print(len(to_del))
        #переприсвоение меняет тот объект, кот создан в конструкторе, а не сам df из параметров
        #если нужно менять df, делаем inplace
        self.df = self.df.drop( to_del, axis = 1) # присвоение не работает

    # удаление фичей с малыми значениями дисперсии (мера разбраса относительно среднего)
    def drop_for_low_var_features(self,trashhold):
        vars = self.df[self.num_features].var()
        drop_features = vars[ vars < trashhold].index.tolist()
        self.df.drop(columns = drop_features, inplace=True)

    def set_features(self):
        current_columns=tuple(self.df.columns)
        if current_columns != self.last_features:
            self.num_features = [col for col in self.df.columns if self.df[col].dtype != 'object' and col!='Hour']
            self.cat_features = [col for col in self.df.columns if col not in self.num_features]
            self.last_features = current_columns



    # энкодинг кат фичей, + числовые с 25 уник  знач
    def encoding_cat_fetures(self, target = None):
        for col in (self.cat_features+[num for num in self.num_features if self.df[num].nunique() < 25]):
            if len(self.df[col].unique()) <= 5:
                self.df = pd.get_dummies(self.df, columns=[col])
            else:
                if target is not None and  target in self.df.columns:
                    encode_col = self.df.groupby(col)[target].mean()
                    self.df[col] = self.df[col].apply(lambda x: encode_col.loc[x])

                # если знаечение для таргета не нашлось, применяем label
                else:
                    mapping  = {val: index for index, val in enumerate(self.df[col].unique())}
                    self.df[col] = self.df[col].map(mapping)

        # Построить график корреляции по тепловой карте и сохранять его в виде png в папку images
    def heat_card_corr(self):
        corr_matrix = self.df[self.num_features].corr().abs()
        os.makedirs("images", exist_ok=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.savefig("images/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()



    def discret_ddm(self, col = 'create_dttm'):
        if col in self.df.columns:
            self.df[col] = pd.to_datetime(self.df[col])
            self.df['Year']=self.df[col].dt.year
            self.df['Month'] = self.df[col].dt.month
            self.df['Day'] = self.df[col].dt.day

    def get_df(self):
        return self.df

    def dtype(self):
        return self.df.dtypes

    # tqdm позволяет отслеживать текущий прогресс и сколько по врмеени осталось до конца итерирования
    def subplots_for_num_features(self, save = None):
        num_cols = len(self.num_features)
        fig, axes  = plt.subplots(nrows= (num_cols // 3) + (num_cols % 3), ncols  = 3, figsize = (16, (num_cols//3 +1)*7))
        for ax, col in tqdm(zip(axes.flatten(), self.num_features), total=num_cols):
            ax.hist(self.df[col], bins  = 30, alpha = 0.7)
            ax.set_title(f'{col} distribution')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.grid()

        for i in range(num_cols, len(axes.flatten())):
            fig.delaxes(axes.flatten()[i])


        if save:
            folder = os.path.dirname(save)
            if folder:
                os.makedirs(folder, exist_ok = True)
            plt.savefig(save, dpi = 300, bbox_inches = 'tight')

        plt.show()



    # построить pairplot в сервисе

    def pairplot_in_df(self):
        sns.pairplot(self.df)
        plt.show()
