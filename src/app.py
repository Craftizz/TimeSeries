from pathlib import Path
from data.preprocessor import DataPreprocessor
from data.loader import load
from data.schema import ColumnSchema
from decompose.visualize import plot
from decompose.decomposer import Decomposer, DecompositionConfig
from forecast.models.arima import ArimaConfig, ArimaForecaster
from forecast.models.prophet import ProphetConfig, ProphetForecaster
from strategy.strategy import StationByProductStrategy, ProductStrategy


if __name__ == "__main__":

    data_directory = Path("data/sales")
    save_directory = Path("results/")

    sales = load(data_directory)

    schema = ColumnSchema()
    strategy = StationByProductStrategy(station=796, product='ADO')

    # arima = ArimaForecaster(
    #     schema=schema,
    #     strategy=strategy,
    #     config=ArimaConfig(validation_days=30)
    # )

    # arima.fit(sales=sales)
    # arima.evaluate()

    # decomposed_data = Decomposer(
    #         schema=schema,
    #         strategy=strategy,
    #         config=DecompositionConfig(),
    #     ).decompose(sales=sales)

    # plot(
    #     decomposed_per_category=decomposed_data,
    #     save_directory=save_directory / strategy.get_folder_name()
    # )

    model = ProphetForecaster(
        schema=schema,
        strategy=strategy,
        config=ProphetConfig(validation_days=30)
    )

    model.fit(sales=sales)
    model.evaluate()

    model.predict()



