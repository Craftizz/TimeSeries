

class DataValidationError(ValueError):
    """Raised when input data fails validation."""
    pass


class InsufficientDataError(ValueError):
    """Raised when data is insufficient for analysis."""
    pass


class ModelNotTrainedError(ValueError):
    """Raised when model is not yet trained or fitted"""
    pass