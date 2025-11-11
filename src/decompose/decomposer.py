from dataclasses import dataclass
from typing import Any

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from data.preprocessor import DataPreprocessor
from data.schema import ColumnSchema
from strategy.strategy import GroupingStrategy


@dataclass(frozen=True)
class DecompositionConfig:

    minimum_data_points: int = 14
    seasonal_period: int = 7
    model_type: str = "additive"
    frequency: str = "D"
    low_sales_std_multiplier = 2.0


@dataclass(frozen=True)
class SalesStatistics:

    mean: float
    std: float
    coefficient_of_variation: float
    low_threshold: float


@dataclass(frozen=True)
class DecompositionResult:

    group_id: dict[str, Any]
    dates: pd.DatetimeIndex
    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    statistics: SalesStatistics
    unusually_low: pd.Series

    def to_dataframe(self) -> pd.DataFrame:

        return pd.DataFrame({

            "Station": self.group_id.get("station"),
            "Group": self.group_id.get("category"),
            "Date": self.dates,
            "Observed": self.observed,
            "Trend": self.trend,
            "Seasonal": self.seasonal,
            "Residual": self.residual,
            "Mean Sales Volume": self.statistics.mean,
            "Std Deviation": self.statistics.std,
            "Coeff of Variation": self.statistics.coefficient_of_variation,
            "Is Unusually Low": self.unusually_low
        })


@dataclass
class Decomposer:

    schema: ColumnSchema
    strategy: GroupingStrategy
    config: DecompositionConfig

    def decompose(self, sales: pd.DataFrame):

        sales = sales.copy()
        sales = DataPreprocessor(
            schema=self.schema, 
            strategy=self.strategy
        ).preprocess(data=sales)

        group_cols = self.strategy.get_grouping_columns(schema=self.schema)
        group_by_cols = [col for col in group_cols if col != self.schema.date]

        results: list[DecompositionResult] = []

        for group_key, group in sales.groupby(group_by_cols):

            # Prepare Time Series
            group = group.set_index(self.schema.date)
            group = group.asfreq(self.config.frequency).fillna(0)

            # Perform Decomposition
            decomposition = seasonal_decompose(
                group[self.schema.sales], 
                model=self.config.model_type, 
                period=self.config.seasonal_period
            )
    
            # Compute Statistics
            mean_sales = group[self.schema.sales].mean()
            std_sales = group[self.schema.sales].std()
            cv_sales = std_sales / mean_sales if mean_sales != 0 else float('nan')

            # Identify unusually low days (below mean - 2*std)
            low_sales_threshold = mean_sales - 2 * std_sales
            unusually_low = group["Sales Vol"] < low_sales_threshold

            # Get group identifier
            group_id = self.strategy.get_group_identifier(group_key)

            results.append(
                DecompositionResult(
                    group_id=group_id,
                    dates=pd.DatetimeIndex(group.index),
                    observed=decomposition.observed,
                    trend=decomposition.trend,
                    seasonal=decomposition.seasonal,
                    residual=decomposition.resid,
                    statistics=SalesStatistics(
                        mean=mean_sales,
                        std=std_sales,
                        coefficient_of_variation=cv_sales,
                        low_threshold=low_sales_threshold,
                    ),
                    unusually_low=unusually_low,
                )
            )

        print(f"Processed {len(results)} groups successfully. Starting to save...")

        return pd.concat(
            [result.to_dataframe() for result in results],
            ignore_index=False
        )