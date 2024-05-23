import numpy.dtypes
import pandas as pd


class DataCleaner:
    def __init__(self, data_frame: pd.DataFrame):
        self.data_frame = data_frame

    def remove_outliers(self):
        for name in self.data_frame:
            series = self.data_frame[name]
            if series.dtype != numpy.dtypes.ObjectDType:
                print(f"{series.name}:")
                mean = series.mean()
                std = series.std()
                print(f"mean: {mean}")
                print(f"std: {std}")
                print(f"min: {series.min()}")
                print(f"max: {series.max()}")
                print(f"{mean - 2 * std, mean + 2 * std}")
