import pandas as pd
from pandas import DataFrame

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from typing import List

from consts import DATASET_PATH, DROP_COLUMNS


class CSVLoader:
    file_path = DATASET_PATH

    @classmethod
    def load_csv(cls) -> DataFrame:
        dataframe = pd.read_csv(filepath_or_buffer=cls.file_path)
        print(dataframe['Сдан оригинал'].value_counts())
        print(dataframe['Приказ о зачислении'].value_counts())
        return dataframe


class Preprocessing(CSVLoader):
    def __init__(self) -> None:
        self.dataframe = self.load_csv()

    def columns(self) -> List[str]:
        columns = self.dataframe.columns
        return columns

    def drop_columns(self) -> None:
        columns = DROP_COLUMNS
        self.dataframe = self.dataframe.drop(
            columns=columns,
            axis=1
        )

    def balance(self) -> None:
        self.dataframe = self.dataframe.sample(frac=1)
        df_true = self.dataframe[
            self.dataframe['Приказ о зачислении'] == 1
        ]
        df_false = self.dataframe[
            self.dataframe['Приказ о зачислении'] == 0
        ].head(df_true.shape[0])
        self.dataframe = pd.concat([df_true, df_false])
        self.dataframe = self.dataframe.sample(frac=1)

    def inputs_outputs_split(self) -> tuple[DataFrame, DataFrame]:
        inputs = self.dataframe.drop('Приказ о зачислении', axis=1)
        outputs = self.dataframe['Приказ о зачислении']
        return inputs, outputs

    @staticmethod
    def train_test_split(
            inputs: DataFrame, outputs: DataFrame
    ) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        x_train, x_test, y_train, y_test = train_test_split(
            inputs, outputs,
            test_size=0.2,
            shuffle=True,
            random_state=1
        )
        return x_train, y_train, x_test, y_test

    @staticmethod
    def normalize(inputs: DataFrame) -> DataFrame:
        columns = inputs.columns
        dataframe_normalize = normalize(inputs)
        inputs = pd.DataFrame(
            data=dataframe_normalize,
            columns=columns
        )
        return inputs

    @staticmethod
    def one_hot_encoding(inputs: DataFrame) -> DataFrame:
        inputs = pd.get_dummies(inputs)
        return inputs


class Dataset(Preprocessing):
    def dataset(self) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        self.drop_columns()
        self.balance()
        x, y = self.inputs_outputs_split()
        x = self.one_hot_encoding(inputs=x)
        x = self.normalize(inputs=x)
        x_train, y_train, x_test, y_test = self.train_test_split(
            inputs=x,
            outputs=y
        )
        print(x_train.shape)
        return x_train, y_train, x_test, y_test
