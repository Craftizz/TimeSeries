from dataclasses import dataclass
import pandas as pd

from data.schema import ColumnSchema
from strategy.strategy import GroupingStrategy
from utils.errors import DataValidationError


@dataclass
class DataPreprocessor:

    schema: ColumnSchema
    strategy: GroupingStrategy
    
    def preprocess(self, 
                   data: pd.DataFrame):

        data = data.copy()

        # Standardadize Product Names
        data[self.schema.product] = (
            data[self.schema.product]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        # Parse Dates
        try:
            data[self.schema.date] = pd.to_datetime(data[self.schema.date])
        except Exception as e:
            raise DataValidationError(
                f"Failed to parse date column '{self.schema.date}': {e}"
            )
        
        # Ensure Numeric Sales
        data[self.schema.sales] = (
            data[self.schema.sales]
            .astype(str)
            .str.replace(r"[^0-9.\-]", "", regex=True)
            .pipe(pd.to_numeric, errors="coerce")
        )

        # Apply strategy filtering
        data = self.strategy.filter_data(data, self.schema)

        group_columns = self.strategy.get_grouping_columns(self.schema)

        data = (
            data
            .groupby(group_columns, as_index=False)
            .agg({self.schema.sales: "sum"})
            .dropna(subset=[self.schema.sales])
            .sort_values(by=self.schema.date)
        )


        return data
