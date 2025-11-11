
from dataclasses import dataclass
from typing import cast

from darts import TimeSeries
from darts.metrics import mae, rmse, mape


@dataclass(frozen=True, slots=True)
class MetricsResult:
    mae: float
    rmse: float
    mape: float

    @classmethod
    def create(cls, 
               actual: TimeSeries, 
               forecast: TimeSeries):
        
        return cls(
            mae=cast(float, mae(actual, forecast)),
            rmse=cast(float, rmse(actual, forecast)),
            mape=cast(float, mape(actual, forecast)),
        )
    
    def display(self) -> None:

        print("\nValidation Metrics:")
        print(f"MAE: {self.mae:.3f}")
        print(f"RMSE: {self.rmse:.3f}")
        print(f"MAPE: {self.mape:.3f}")