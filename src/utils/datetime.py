import pandas as pd

def daterange_business_days(start: str, end: str) -> list[str]:
    """Return business days (Mon-Fri) between start and end inclusive."""
    dates = pd.date_range(start=start, end=end, freq="B")
    return [d.strftime("%Y-%m-%d") for d in dates]