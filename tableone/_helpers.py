import re
import warnings
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats

_category, _value, _paren = _columns = ["category", "value", "paren"]


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


def prettify(df: pd.DataFrame,
             name: Union[str, List[str]] = "") -> pd.DataFrame:
    """Prettify a dataframe

    For example::
        >>> df = pd.DataFrame([["Mean data (SD)", 1.256, 0.254],
        ...                    ["Median data (IQR)", 2.564, 2.543]]])
        >>> prettify(df, name="Statistic)
        category            Statistic
        Mean data (SD)      1.26 (0.25)
        Median data (IQR)   2.56 (2.54)
        >>> df = pd.DataFrame([["Count 1", 10, 0.0314159],
        ...                    ["Count 2", 20, 0.0271828]])
        >>> prettify(df, name="Count")
        category  Count
        Count 1    10 (3.14%)
        Count 2    20 (2.72%)
        >>> df = pd.DataFrame([["Mean data1 (95% CI)", 1.256, (0.523, 1.523)],
        ...                    ["Mean data2 (95% CI)", 2.564, (1.523, 2.523)]])
        >>> prettify(df, name="Statistic")
        category             Statistic
        Mean data1 (95% CI)  1.26 (0.52, 1.52)
        Mean data2 (95% CI)  2.56 (1.52, 2.52)
        >>> df = pd.DataFrame([["Count 1", 10, 0.0314159, 20, 0.0271828],
        ...                    ["Count 2", 20, 0.0271828, 30, 0.0245283]])
        >>> prettify(df, name=["Count1", "Count2"])
        category   Count1      Count2
        Count 1    10 (3.14%)  20 (2.72%)
        Count 2    20 (2.72%)  30 (2.45%)

    :param df: The dataframe to prettify
    :type df: pd.DataFrame
    :param name: The name or names of the output column or columns
    :type name: str or list of str

    :return: The prettified dataframe
    :rtype: pd.DataFrame
    """
    out = df.set_index(df.columns[0])
    if not isinstance(name, str) and len(name) != len(out.columns) / 2:
        raise ValueError("name must be half in length as the number of "
                         f"columns. Instead, got {len(name)} names "
                         f"and {len(out.columns)} columns.")
    if isinstance(name, str):
        name = [name]

    # To apply row by row
    def print_proper(row: pd.Series) -> pd.DataFrame:
        printed = pd.Series(dtype="object", index=name)
        for i, n in enumerate(name):
            val = row.iloc[2 * i + 0]
            paren = row.iloc[2 * i + 1]
            if row.eq("").all() or row.isna().all():
                printed[n] = ""
                continue
            if isinstance(val, float):
                val = f"{val:.2f}"

            if isinstance(paren, str):
                printed[n] = f"{val} ({paren})"
            elif isinstance(paren, tuple):
                # Confidence intervals
                printed[n] = f"{val} ({paren[0]:.2f}, {paren[1]:.2f})"
            elif re.match(r"Me(?:di)?an .* \(.+\)", row.name):
                # Means and medians
                printed[n] = f"{val} ({paren:.2f})"
            else:
                # Percentages
                printed[n] = f"{float(val):.0f} ({paren:.2%})"
        return printed

    out[name] = out.apply(print_proper, axis=1)
    out = out.reset_index()[[_category] + name]
    return out


def numerical_calculation(col: Union[pd.Series, List[pd.Series], pd.DataFrame],
                          val_func: Callable,
                          spread_func: Callable,
                          text: str = None,
                          as_str: bool = False,
                          name: str = "") -> pd.DataFrame:
    """Calculate a numerical statistic for a column

    For example::
        >>> col = pd.Series([1, 2, 3, 4, 5], name="A")
        >>> numerical_calculation(col, np.mean, np.std)
        category      value  paren
        Mean A (std)  2.50   0.50
        >>> numerical_calculation(col, np.mean, np.std, as_str=True)
        category      Mean
        Mean A (std)  2.50 (0.50)

    :param col: The column to calculate the statistic for
    :type col: pd.Series or list of pd.Series
    :param val_func: The function to calculate the value
    :type val_func: Callable
    :param spread_func: The function to calculate the spread
    :type spread_func: Callable
    :param text: The name of the resulting row. Defaults to
        ``{val_func.__name__} {col.name} ({spread_func.__name__})``
    :type text: str
    :param as_str: Whether to return the output as a dataframe of strings.
        Defaults to ``False``.
    :type as_str: bool
    :param name: The name of the resulting column. Defaults to ``""``.
    :type name: str

    :return: The calculated statistic dataframe
    :rtype: pd.DataFrame
    """
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
    """Return a dataframe with the name and data

    For example::
        >>> _to_dataframe("Mean", (1.0, 0.1))
        category  value  paren
        Mean      1.00   0.10

    :param name: The name of the row
    :type name: str
    :param data: The data to put in the row
    :type data: Tuple[float, float]

    :return: A dataframe with the name and data
    :rtype: pd.DataFrame
    """
    if len(data) > 2:
        warnings.warn(f"Too many data points provided for {name} and the "
                      "extras will be dropped.")
    return pd.DataFrame([[name, data[0], data[1]]], columns=_columns)


def categorical_calculation(col: Union[pd.Series, List[pd.Series],
                                       pd.DataFrame],
                            as_str: bool = False,
                            name: str = "") -> pd.DataFrame:
    """Counts the number of times each category occurs in a column

    For example::
        >>> col = pd.Series(["a", "b", "a", "b", "a", "b"])
        >>> categorical_calculation(col)
        category  value  paren
        a         2      0.50
        b         2      0.50
        >>> categorical_calculation(col, as_str=True, name="Count")
        category  Count
        a         2 (50.00%)
        b         2 (50.00%)

    :param col: The column or columns to count in. If a series, just use the
        series. If a list of series, concatenate the outputs. If a dataframe,
        concatenate the outputs of each column
    :type col: pd.Series or List[pd.Series] or pd.DataFrame
    :param as_str: Whether to return the data as a dataframe of strings or a
        dataframe of floats.
    :type as_str: bool
    :param name: The name of the output columns.

    :return: A dataframe of the counts.
    :rtype: pd.DataFrame
    """
    if isinstance(col, pd.DataFrame):
        return pd.concat([
            categorical_calculation(col[c], as_str=as_str, name=name)
            for c in col.columns
        ]).reset_index(drop=True)

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
    out[_category] = out[_category].astype(str)
    out = out.sort_values([_value, _category], ascending=(False, True))
    out = pd.concat([top_row, out]).reset_index(drop=True)

    if as_str:
        return prettify(out, name=name)
    return out


def chi_square(data: np.ndarray) -> tuple:
    expect = (data.sum(axis=0) * data.sum(axis=1).reshape(-1, 1)) / data.sum()
    chi = scipy.stats.chisquare(data, expect, axis=None)

    return chi


def ttest(column: pd.Series, condition: pd.Series) -> Tuple[float, float]:
    trues = column[condition].values
    falses = column[~condition].values
    return scipy.stats.ttest_ind(trues, falses)


def median_test(column: pd.Series, condition: pd.Series):
    trues = column[condition].values
    falses = column[~condition].values
    return scipy.stats.median_test(trues, falses)
