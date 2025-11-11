
from dataclasses import dataclass, field

import pandas as pd
from darts.models import Prophet
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

from forecast.data.transformer_pipeline import DataSplit, DataTransformer
from forecast.evaluation.metrics import MetricsResult
from forecast.models.base_forecaster import BaseForecaster
from forecast.models.base_config import BaseConfig
from utils.errors import ModelNotTrainedError


@dataclass
class ProphetConfig(BaseConfig):

    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = True
    seasonality_mode: str = "additive"
    change_prior_scale: float = 0.05
    checkpoint_range: float = 0.9


@dataclass
class ProphetForecaster(BaseForecaster):

    config: ProphetConfig

    model: Prophet = field(init=False)
    transformer: DataTransformer = field(init=False)
    datasplit: DataSplit = field(init=False)

    def __post_init__(self) -> None:

        self.transformer = DataTransformer(config=self.config, 
                                           schema=self.schema, 
                                           strategy=self.strategy)
        
        self.model = Prophet(yearly_seasonality=self.config.yearly_seasonality,
                             weekly_seasonality=self.config.weekly_seasonality,
                             daily_seasonality=self.config.daily_seasonality,
                             seasonality_mode=self.config.seasonality_mode,
                             changepoint_prior_scale=self.config.change_prior_scale,
                             changepoint_range=self.config.checkpoint_range)
    
    def fit(self, sales: pd.DataFrame) -> None:

        self.datasplit = self.transformer.transform(sales)

        print("Starting Prophet training...")
        self.model.fit(self.datasplit.train)
        print("Training complete!")

    
    def evaluate(self) -> None:

        if self.model is None:
            raise ModelNotTrainedError("Model must be fitted before evaluating")
        
        forecast = self.model.predict(len(self.datasplit.val))

        forecast = self.transformer.inverse(forecast)
        train = self.transformer.inverse(self.datasplit.train)
        val = self.transformer.inverse(self.datasplit.val)

        metrics = MetricsResult.create(actual=val, forecast=forecast)
        metrics.display()

        plt.figure(figsize=(10, 5))
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style="plain", axis="y")

        train.plot(label="Train")
        val.plot(label="Validation (Actual)")
        forecast.plot(label="Forecast", lw=2)

        plt.legend()
        plt.title("Prophet Train / Validation Forecast Comparison")
        plt.show()


    def predict(self):
        pass