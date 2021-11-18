"""For the Table One class"""
import re
import warnings
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd

from ._helpers import ci as ci_
from ._helpers import iqr


class TableOne():
    _category, _value, _paren = _columns = ["category", "value", "paren"]

    def __init__(self,
                 data: pd.DataFrame,
                 categorical: list,
                 numerical: list,
                 groupings: list = None):
        self.data = data.copy()

        # Lowercase all column names
        def lower(x):
            return x.lower()

        self.data.columns = map(lower, self.data.columns)
        self.cat = list(map(lower, categorical))
        self.num = list(map(lower, numerical))
        missing = (set(self.cat) | set(self.num)) - set(self.data.columns)
        if len(missing) > 0:
            raise ValueError(f"Missing columns: {missing}")
        # Remove duplicates between cat and num while maintaining order
        # A nifty trick I pulled from StackOverflow (https://bit.ly/3DbcgoI)
        columns = list(dict.fromkeys(self.cat + self.num))

        self.data = self.data[columns]
        self.n = len(self.data)

        # This is coming next
        # self.groupings = [] if groupings is None else groupings
        # if missing := (set(self.groupings) - set(self.data.columns)):
        #     raise ValueError("Groupings must be columns in data."
        #                      f"Missing {missing}")

    def __repr__(self):
        return f"TableOne({self.n} patients)"

    def __str__(self) -> str:
        return f"TableOne({self.n} patients)"

    @staticmethod
    def _to_dataframe(name: str, data: Tuple[float, float]) -> pd.DataFrame:
        """Return a dataframe with the name and data"""
        if len(data) > 2:
            warnings.warn(f"Too many data points provided for {name} and the "
                          "extras will be dropped.")
        return pd.DataFrame([[name, data[0], data[1]]],
                            columns=TableOne._columns)

    def mean_and_sd(self,
                    col: Union[str, list] = None,
                    as_str: bool = False) -> pd.DataFrame:
        """Return the mean and standard deviation of columns

        :param col: The column or columns to analyze.
        :type col: str or list

        :returns: A dataframe with the mean and standard deviation of the
            column.
        """
        def mean(col: pd.Series) -> float:
            """Return the mean of a column"""
            return col.mean()

        mean.__name__ = "Mean"

        def sd(col: pd.Series) -> float:
            """Return the standard deviation of a column"""
            return col.std()

        sd.__name__ = "SD"

        return self._num_calc(col, mean, sd, as_str=as_str)

    def median_and_iqr(self,
                       col: Union[str, list] = None,
                       as_str: bool = False) -> pd.DataFrame:
        """Return the median and interquartile range of columns

        :param col: The column or columns to analyze.
        :type col: str or list

        :returns: A dataframe with the median and interquartile range of the
            column.
        """
        def median(col: pd.Series) -> float:
            """Return the median of a column"""
            return col.median()

        median.__name__ = "Median"

        def _iqr(col: pd.Series) -> float:
            """Return the interquartile range of a column"""
            return iqr(col)

        _iqr.__name__ = "IQR"

        return self._num_calc(col, median, _iqr, as_str=as_str)

    def mean_and_ci(self,
                    col_name: Union[str, list] = None,
                    as_str: bool = False) -> pd.DataFrame:
        """Mean and 95% confidence interval"""

        # Just for naming purposes
        def _mean(col: pd.Series) -> float:
            """Return the mean of a column"""
            return col.mean()

        _mean.__name__ = "Mean"

        def _ci(col: pd.Series) -> Tuple[float, float]:
            """Return the 95% confidence interval of a column"""
            return ci_(col)

        _ci.__name__ = "95% CI"

        return self._num_calc(col_name, _mean, _ci, as_str=as_str)

    def _num_calc(self,
                  col_name: Union[str, List],
                  val_func: Callable,
                  spread_func: Callable,
                  text: str = None,
                  as_str: bool = False) -> pd.DataFrame:
        if col_name is None:
            return self._num_calc(self.num,
                                  val_func,
                                  spread_func,
                                  text=text,
                                  as_str=as_str)
        if not isinstance(col_name, str):
            return pd.concat([
                self._num_calc(c,
                               val_func,
                               spread_func,
                               text=text,
                               as_str=as_str) for c in col_name
            ]).reset_index(drop=True)

        col_name = col_name.lower()
        col = self.data[col_name]

        if not col_name in self.num:
            warnings.warn(f"{col_name} is not a numeric column.")
            col = pd.to_numeric(col, errors="coerce")

        val = val_func(col)
        spread = spread_func(col)
        if text is None:
            text = (f"{val_func.__name__} {col_name} ({spread_func.__name__})")

        out = self._to_dataframe(text, (val, spread))

        if as_str:
            return self.prettify(out)
        return out

    def counts(self,
               col_name: Union[str, list] = None,
               as_str: bool = False) -> pd.DataFrame:
        """Return the counts of a column"""
        if col_name is None:
            return self.counts(self.cat, as_str=as_str)

        if not isinstance(col_name, str):
            return pd.concat([self.counts(c, as_str=as_str)
                              for c in col_name]).reset_index(drop=True)

        col_name = col_name.lower()
        if not col_name in self.cat:
            warnings.warn(f"{col_name} is not a categorical column.")

        col = self.data[col_name]
        # The counts, not normalized
        unnormed = col.value_counts(dropna=False)
        unnormed.name = self._value
        unnormed.index.name = self._category

        # The counts, normalized
        normed = unnormed / self.n
        normed.name = self._paren
        normed.index.name = self._category

        top_row = pd.DataFrame([[col_name, np.nan, np.nan]],
                               columns=self._columns)
        out = unnormed.to_frame().join(normed.to_frame()).reset_index()
        out[self._category] = out[self._category].astype(str)
        out = out.sort_values([self._value, self._category],
                              ascending=(False, True))
        out = pd.concat([top_row, out]).reset_index(drop=True)

        if as_str:
            return self.prettify(out)
        return out

    def count_na(self) -> pd.Series:
        """Return the number of missing values per column"""
        return self.data.isna().sum()

    def id_invalid(self) -> dict:
        """Identify the invalid values in the numeric columns"""
        out = dict()
        for column in self.num:
            try:
                pd.to_numeric(self.data[column], errors="raise")
            except ValueError:
                coerced = pd.to_numeric(self.data[column], errors="coerce")
                # Values that can't be coerced to numeric
                new_na = coerced[coerced.isna()
                                 & ~self.data[column].isna()].index
                out[column] = list(new_na)

        return out

    def drop_invalid(self) -> None:
        """Drop the invalid values in the numeric columns"""
        invalid = self.id_invalid()
        for column in invalid:
            self.data.loc[invalid[column], column] = np.nan

    @staticmethod
    def prettify(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
        out = df.set_index(df.columns[0])

        # To apply row by row
        def print_proper(row: pd.Series) -> str:
            val = row.iloc[0]
            paren = row.iloc[1]
            if row.isna().all():
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
        out = out.reset_index()[[TableOne._category, name]]
        return out

    def analyze_categorical(self, as_str: bool = False) -> pd.DataFrame:
        out = self.counts(self.cat, as_str=as_str).reset_index(drop=True)
        return out

    def analyze_numeric(self, as_str: bool = False) -> pd.DataFrame:
        if invalid := self.id_invalid():
            raise ValueError("Invalid values found in numeric columns: "
                             f"{invalid}")
        out = pd.concat([
            self.mean_and_sd(self.num, as_str=as_str),
            self.mean_and_ci(self.num, as_str=as_str),
            self.median_and_iqr(self.num, as_str=as_str)
        ]).reset_index(drop=True)
        return out

    def analyze(self, as_str: bool = False) -> pd.DataFrame:
        out = pd.concat([
            self.analyze_categorical(as_str=as_str),
            self.analyze_numeric(as_str=as_str)
        ]).reset_index(drop=True)
        return out
