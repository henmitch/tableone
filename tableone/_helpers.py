import re
import warnings
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats

_category, _value, _paren = _columns = ["category", "value", "paren"]

_bool_conversion = {
    "TRUE": True,
    "FALSE": False,
    "YES": True,
    "NO": False,
    "Y": True,
    "N": False,
    "1": True,
    "0": False,
    "T": True,
    "F": False
}


def ci(ser: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
    """The confidence interval for the mean of a series

    :ser: The series of which to find the mean/CI.
    :type ser: pd.Series
    :confidence: The confidence to use.
    :type confidence: float

    :return: The lower and upper bounds of the confidence interval.
    """
    mean = ser.mean()
    n = ser.size
    df = n - 1
    t_score = scipy.stats.t.interval(confidence, df=df)
    low = mean + t_score[0] * ser.std() / np.sqrt(n)
    high = mean + t_score[1] * ser.std() / np.sqrt(n)
    return (low, high)


def iqr(ser: pd.Series) -> float:
    """The interquartile range of a series"""
    return ser.quantile(0.75) - ser.quantile(0.25)


def prettify(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    out = df.set_index(df.columns[0])

    # To apply row by row
    def print_proper(row: pd.Series) -> str:
        val = row.iloc[0]
        paren = row.iloc[1]
        if row.eq("").all() or row.isna().all():
            return ""
        if isinstance(val, float):
            val = f"{val:.2f}"

        if isinstance(paren, str):
            return f"{val} ({paren})"
        # Confidence intervals
        if isinstance(paren, tuple):
            return f"{val} ({paren[0]:.2f}, {paren[1]:.2f})"
        # Means and medians
        if re.match(r"Me(?:di)?an .* \(.+\)", row.name):
            return f"{val} ({paren:.2f})"
        # Percentages
        return f"{float(val):.0f} ({paren:.2%})"

    out[name] = out.apply(print_proper, axis=1)
    out = out.reset_index()[[_category, name]]
    return out


def numerical_calculation(col: Union[pd.Series, List[pd.Series], pd.DataFrame],
                          val_func: Callable,
                          spread_func: Callable,
                          text: str = None,
                          as_str: bool = None,
                          name: str = "") -> pd.DataFrame:
    if isinstance(col, pd.DataFrame):
        return pd.concat([
            numerical_calculation(col[c],
                                  val_func,
                                  spread_func,
                                  text,
                                  as_str,
                                  name=name) for c in col.columns
        ]).reset_index(drop=True)

    val = val_func(col)
    spread = spread_func(col)
    if text is None:
        text = f"{val_func.__name__} {col.name} ({spread_func.__name__})"

    out = _to_dataframe(text, (val, spread))

    if as_str:
        return prettify(out, name=name)
    return out


def _to_dataframe(name: str, data: Tuple[float, float]) -> pd.DataFrame:
    """Return a dataframe with the name and data"""
    if len(data) > 2:
        warnings.warn(f"Too many data points provided for {name} and the "
                      "extras will be dropped.")
    return pd.DataFrame([[name, data[0], data[1]]], columns=_columns)


def categorical_calculation(col: Union[pd.Series, List[pd.Series]],
                            as_str: bool = False,
                            name: str = "") -> pd.DataFrame:
    n = col.size
    unnormed = col.value_counts(dropna=False)
    unnormed.name = _value
    unnormed.index.name = _category

    # The counts, normalized
    normed = unnormed / n
    normed.name = _paren
    normed.index.name = _category

    top_row = pd.DataFrame([[col.name.capitalize(), "", ""]], columns=_columns)
    out = unnormed.to_frame().join(normed.to_frame()).reset_index()
    out[_category] = "    " + out[_category].astype(str)
    out = out.sort_values([_value, _category], ascending=(False, True))
    out = pd.concat([top_row, out]).reset_index(drop=True)

    if as_str:
        return prettify(out, name=name)
    return out


def booleanize(col: pd.Series, fillna: bool = None) -> pd.Series:
    """Convert a series to a boolean series

    :col: The series to convert.
    :type col: pd.Series

    :return: A boolean series.
    :rtype: pd.Series
    """
    try:
        col = pd.to_numeric(col, errors="raise")
    except (ValueError, TypeError):
        pass

    if np.issubdtype(col.dtype, np.number):
        out = (col > 0)
    elif col.dtype == object:
        out = col.str.upper().apply(lambda x: _bool_conversion.get(x, None))
    else:
        out = col

    out[col.isna()] = None
    if fillna is not None:
        out = out.fillna(fillna)

    return out
