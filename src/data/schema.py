from dataclasses import dataclass

@dataclass(frozen=True)
class ColumnSchema:

    date: str = "Transaction Date"
    sales: str = "Sales Vol"
    product: str = "Product"
    station: str = "Station #"
