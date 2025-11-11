from dataclasses import dataclass, field
from typing import Sequence

from darts import TimeSeries
import pandas as pd
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from sklearn.preprocessing import RobustScaler

from data.preprocessor import DataPreprocessor
from data.schema import ColumnSchema
from forecast.models.base_config import BaseConfig
from strategy.strategy import GroupingStrategy


@dataclass(frozen=True, slots=True)
class DataSplit:
    train: TimeSeries
    val: TimeSeries
    
    @property
    def train_size(self) -> int:
        return len(self.train)
    
    @property
    def val_size(self) -> int:
        return len(self.val)
    
    def display_info(self) -> None:
        print(f"Training on {self.train_size} points | "
              f"Validating on {self.val_size} points")


@dataclass(slots=True)
class DataFramePreprocessor:

    schema: ColumnSchema
    strategy: GroupingStrategy
    
    def preprocess(self, 
                   sales: pd.DataFrame) -> pd.DataFrame:
        
        sales = sales.copy()
        
        return DataPreprocessor(
            schema=self.schema,
            strategy=self.strategy
        ).preprocess(data=sales)


@dataclass(slots=True)
class TimeSeriesBuilder:

    schema: ColumnSchema
    config: BaseConfig

    filler: MissingValuesFiller = MissingValuesFiller(fill='auto')

    def build(self, data: pd.DataFrame) -> TimeSeries:

        # Create to Time Series
        series = TimeSeries.from_dataframe(
            data,
            time_col=self.schema.date,
            value_cols=self.schema.sales,
            fill_missing_dates=self.config.fill_missing_dates,
            freq=self.config.frequency
        )

        # Fill Missing Values
        filled = self.filler.transform(series)
        return filled if isinstance(filled, TimeSeries) else filled[0]
    

    def split(self, series: TimeSeries) -> DataSplit:
        return DataSplit(
            train=series[:-self.config.validation_days],
            val=series[-self.config.validation_days:]
        )


@dataclass(slots=True)
class SeriesScaler:
    
    scaler: Scaler = Scaler(RobustScaler())

    def _ensure_single_series(self, result: object) -> TimeSeries:
        if isinstance(result, TimeSeries):
            return result

        if isinstance(result, (list, tuple)):
            current = result
            while True:
                if isinstance(current, TimeSeries):
                    return current
                if isinstance(current, (list, tuple)) and len(current) > 0:
                    current = current[0]
                    continue
                raise TypeError(f"Cannot extract TimeSeries from object of type {type(result)!r}")

        raise TypeError(f"Cannot extract TimeSeries from object of type {type(result)!r}")
    
    def scale(self, split: DataSplit) -> DataSplit:

        train_scaled = self._ensure_single_series(self.scaler.fit_transform(split.train))
        val_scaled = self._ensure_single_series(self.scaler.transform(split.val))

        return DataSplit(train=train_scaled, val=val_scaled)

    def inverse(self, series: TimeSeries) -> TimeSeries:

        inversed = self.scaler.inverse_transform(series)
        return self._ensure_single_series(inversed)



@dataclass(slots=True)
class DataTransformer:

    config: BaseConfig
    schema: ColumnSchema
    strategy: GroupingStrategy

    builder: TimeSeriesBuilder = field(init=False)
    scaler: SeriesScaler = field(init=False)
    datasplit: DataSplit = field(init=False)
    

    def __post_init__(self) -> None:
        self.builder = TimeSeriesBuilder(self.schema, self.config)
        self.scaler = SeriesScaler()
    

    def transform(self, sales: pd.DataFrame) -> DataSplit:

        # Preprocess and Standardize Data
        sales = sales.copy()
        sales = DataPreprocessor(
            schema=self.schema, strategy=self.strategy
        ).preprocess(data=sales)

        # Create Series from Data
        series = self.builder.build(sales)

        # Split Data and Display Info
        self.datasplit = self.builder.split(series)
        self.datasplit.display_info()

        # Scale Data
        split = self.scaler.scale(self.datasplit)

        return split
    

    def inverse(self, series: TimeSeries) -> TimeSeries:

        return self.scaler.inverse(series)
        