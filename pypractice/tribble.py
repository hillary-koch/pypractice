"""
Equivalent to R's tidyr::tribble; a clever use of iterators
"""
import pandas as pd

def tribble(columns, *data):
    return pd.DataFrame(
        data=list(zip(*[iter(data)]*len(columns))),
        columns=columns
    )