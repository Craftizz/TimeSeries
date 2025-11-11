from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import pandas as pd

from data.schema import ColumnSchema
from utils.errors import DataValidationError


class GroupingStrategy(ABC):

    @abstractmethod
    def get_category_column(self, 
                            schema: ColumnSchema) -> str:
        pass

    @abstractmethod
    def get_grouping_columns(self, 
                             schema: ColumnSchema) -> list[str]:
        pass
    
    @abstractmethod
    def filter_data(self, 
                    df: pd.DataFrame, 
                    schema: ColumnSchema) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def get_group_identifier(self, 
                             group_key: tuple) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_folder_name(self) -> str:
        pass

@dataclass
class ProductStrategy(GroupingStrategy):

    product: Optional[str] = None

    def get_category_column(self, 
                            schema: ColumnSchema):
        
        return schema.product

    def get_grouping_columns(self, 
                             schema: ColumnSchema):
         
         return [schema.product, schema.date]

    def filter_data(self, 
                    data: pd.DataFrame, 
                    schema: ColumnSchema):
        
        if self.product is not None:
            data = data[data[schema.product] == self.product]

        if data.empty:
            raise DataValidationError(
                f"No data found for Product {self.product}"
            )
        
        return data

        return data

    def get_group_identifier(self, 
                             group_key: tuple):
        
        category = group_key[0] if isinstance(group_key, tuple) else group_key

        return {
            "category": category, 
            "station": None
        }
    
    def get_folder_name(self):
        return 'per_product'

@dataclass
class StationByProductStrategy(GroupingStrategy):
    
    station: Optional[int] = None
    product: Optional[str] = None

    def get_category_column(self, 
                            schema: ColumnSchema):
        
        return schema.product

    def get_grouping_columns(self, 
                             schema: ColumnSchema):
        
        return [schema.station, schema.product, schema.date]
    
    def filter_data(self, 
                    data: pd.DataFrame, 
                    schema: ColumnSchema):
        
        
        if self.station is not None:
            data = data[data[schema.station] == self.station]

        if self.product is not None:
            data = data[data[schema.product] == self.product]

        if data.empty:
            raise DataValidationError(
                f"No data found for Product {self.product} on Station {self.station}"
            )
        
        return data
    
    def get_group_identifier(self, group_key: tuple):

        if isinstance(group_key, tuple) and len(group_key) >= 2:

            return {
                "station": group_key[0], 
                "category": group_key[1]
            }
        
        return {
            "station": None, 
            "category": group_key
        }
    
    def get_folder_name(self):

        if self.station is None:
            raise ValueError("Station cannot be None")
        
        return Path("per_station") / f"station {self.station}"