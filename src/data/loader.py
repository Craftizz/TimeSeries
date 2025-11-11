from pathlib import Path
import pandas as pd


def load(directory: Path):

    dataset = []

    for file in directory.glob("*.csv"):
        dataframe = pd.read_csv(file)
        dataframe.columns = dataframe.columns.str.strip()

        dataset.append(dataframe)

    return pd.concat(dataset, ignore_index=True)
