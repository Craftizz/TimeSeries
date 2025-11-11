from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from data.schema import ColumnSchema
from forecast.models.base_config import BaseConfig
from strategy.strategy import GroupingStrategy

@dataclass
class BaseForecaster(ABC):

    schema: ColumnSchema
    strategy: GroupingStrategy
    config: BaseConfig

    @abstractmethod
    def fit(self, sales: pd.DataFrame):
        pass
    

    @abstractmethod
    def evaluate(self):
        pass


    @abstractmethod
    def predict(self, days: int):
        pass


        