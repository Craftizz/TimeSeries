

from dataclasses import dataclass, field

from darts.models import AutoARIMA
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd

from forecast.data.transformer_pipeline import DataSplit, DataTransformer
from forecast.evaluation.metrics import MetricsResult
from forecast.models.base_forecaster import BaseForecaster
from forecast.models.base_config import BaseConfig
from utils.errors import ModelNotTrainedError

@dataclass
class ArimaConfig(BaseConfig):

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

@dataclass
class ArimaForecaster(BaseForecaster):

    config: ArimaConfig

    model: AutoARIMA = field(init=False)
    transformer: DataTransformer = field(init=False)
    datasplit: DataSplit = field(init=False)

    def __post_init__(self) -> None:

        self.transformer = DataTransformer(config=self.config, 
                                           schema=self.schema, 
                                           strategy=self.strategy)

        self.model = AutoARIMA(start_p=self.config.start_p,
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
                               approximation=self.config.approximation)


    def fit(self, sales: pd.DataFrame) -> None:

        self.datasplit = self.transformer.transform(sales)

        print("Starting Arima training...")
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
        plt.title("Arima Train / Validation Forecast Comparison")
        plt.show()

    def predict(self) -> None:

        pass