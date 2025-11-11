from dataclasses import dataclass

@dataclass
class BaseConfig:
    frequency: str = "D"
    fill_missing_dates: bool = True
    validation_days: int = 30



