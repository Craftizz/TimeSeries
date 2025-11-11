from dataclasses import dataclass, field
from typing import Optional, cast

from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import AutoARIMA
from darts.metrics import mae, rmse, mape
from darts.dataprocessing.transformers import Scaler, BoxCox, MissingValuesFiller
from sklearn.preprocessing import RobustScaler
from scipy.stats.mstats import winsorize
from scipy.stats import iqr

from data.preprocessor import DataPreprocessor
from data.schema import ColumnSchema
from strategy.strategy import GroupingStrategy
from utils.errors import ModelNotTrainedError


@dataclass
class ModelConfig:

    frequency: str = 'D'
    fill_missing_dates: bool = True

    start_p: int = 2
    start_q: int = 2
    max_p: int = 7
    max_q: int = 7
    max_P: int = 2
    max_Q: int = 2
    max_D: int = 1
    d: int | None = None
    D: int | None = None
    seasonal: bool = True
    season_length: int = 7
    stepwise: bool = False
    approximation: bool = False
    trace: bool = True

    validation_days: int = 30   


@dataclass(frozen=True, slots=True)
class MetricsResult:
    mae: float
    rmse: float
    mape: float
    
    def display(self) -> None:
        print("\nValidation Metrics:")
        print(f"MAE: {self.mae:.3f}")
        print(f"RMSE: {self.rmse:.3f}")
        print(f"MAPE: {self.mape:.3f}")






@dataclass
class ArimaForecaster:

    schema: ColumnSchema
    strategy: GroupingStrategy
    config: ModelConfig
    model: Optional[AutoARIMA] = field(init=False, default=None)
    series: Optional[TimeSeries] = field(init=False, default=None)
    train: Optional[TimeSeries] = field(init=False, default=None)
    val: Optional[TimeSeries] = field(init=False, default=None)

    scaler: Scaler = Scaler(RobustScaler())

    def fit(self, sales: pd.DataFrame):

        sales = sales.copy()
        sales = DataPreprocessor(
            schema=self.schema, 
            strategy=self.strategy
        ).preprocess(data=sales)

        series = TimeSeries.from_dataframe(
            sales,
            time_col=self.schema.date,
            value_cols=self.schema.sales,
            fill_missing_dates=self.config.fill_missing_dates, 
            freq=self.config.frequency
        )

        filler = MissingValuesFiller(fill='auto')
        series = filler.transform(series=series) 


        train, val = (
            series[: -self.config.validation_days],
            series[-self.config.validation_days :],
        )
        print(f"Training on {len(train)} points " 
              f"Validating on {len(val)} points ")
        
        train = self.scaler.fit_transform(train)
        val = self.scaler.transform(val)

        model = AutoARIMA(
            start_p=self.config.start_p,
            start_q=self.config.start_q,
            max_p=self.config.max_p,
            max_q=self.config.max_q,
            max_P=self.config.max_P,
            max_Q=self.config.max_Q,
            max_D=self.config.max_D,
            d=self.config.d,
            D=self.config.D,
            seasonal=self.config.seasonal,
            season_length=self.config.season_length,
            stepwise=self.config.stepwise,
            trace=self.config.trace,
            approximation=self.config.approximation
        )

        print("Starting model training...")
        model.fit(train)
        print("Training complete!")

        self.model = model
        self.series = series
        self.train = train
        self.val = val

    def evaluate(self):

        if self.model is None or self.val is None:
            raise ModelNotTrainedError("Model must be fitted before evaluating")

        forecast = self.model.predict(len(self.val))

        forecast_inversed = self.scaler.inverse_transform(forecast)
        train_inversed = self.scaler.inverse_transform(self.train)
        val_inversed = self.scaler.inverse_transform(self.val)

        metrics = MetricsResult(
            mae=cast(float, mae(val_inversed, forecast_inversed)),
            rmse=cast(float, rmse(val_inversed, forecast_inversed)),
            mape=cast(float, mape(val_inversed, forecast_inversed))
        )

        metrics.display()

        plt.figure(figsize=(10, 5))

        ax = plt.gca()
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='y')

        train_inversed.plot(label="Train")
        val_inversed.plot(label="Validation (Actual)")
        forecast_inversed.plot(label="Forecast", lw=2)

        plt.legend()
        plt.title("Train / Validation Forecast Comparison")
        plt.show()











    def predict(self, days: int = 30):

        if self.model is None:
            raise ModelNotTrainedError("Model must be fitted before prediction")

        forecast = self.model.predict(days)


        forecast_inversed = self.scaler.inverse_transform(forecast)

        forecast_inversed.plot(label="Forecast", lw=2)
        # self.series.plot(label="Historical")

        # self.series.plot(label="Historical")
        # forecast.plot(label="Forecast", lw=2)
        # print(forecast)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='y')

        plt.legend()
        plt.show()

        