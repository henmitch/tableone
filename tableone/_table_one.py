"""For the Table One class"""
import warnings
from typing import Tuple, Union

import numpy as np
import pandas as pd

from ._helpers import ci as ci_


class TableOne():
    _columns = ["category", "value", "paren"]

    def __init__(self, data: pd.DataFrame, categorical: list, numerical: list):
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
        # Remove duplicates between cat and num
        # A nifty trick I pulled from StackOverflow (https://bit.ly/3DbcgoI)
        columns = list(dict.fromkeys(self.cat + self.num))

        self.data = self.data[columns]
        self.n = len(self.data)
        # A default value

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

    def mean_and_sd(self, col: Union[str, list] = None) -> pd.DataFrame:
        """Return the mean and standard deviation of columns

        :param col: The column or columns to analyze.
        :type col: str or list

        :returns: A dataframe with the mean and standard deviation of the
            column.
        """
        if col is None:
            return self.mean_and_sd(self.num)
        if not isinstance(col, str):
            return pd.concat([self.mean_and_sd(c) for c in col])

        col = col.lower()
        if not col in self.num:
            warnings.warn(f"{col} is not a numeric column.")
        mean = self.data[col].mean()
        sd = self.data[col].std()
        text = f"Mean {col} (SD)"
        return self._to_dataframe(text, (mean, sd))

    def median_and_iqr(self, col: Union[str, list] = None) -> pd.DataFrame:
        """Return the median and interquartile range of columns

        :param col: The column or columns to analyze.
        :type col: str or list

        :returns: A dataframe with the median and interquartile range of the
            column.
        """
        if col is None:
            return self.median_and_iqr(self.num)
        if not isinstance(col, str):
            return pd.concat([self.median_and_iqr(c) for c in col])

        col = col.lower()
        if not col in self.num:
            warnings.warn(f"{col} is not a numeric column.")
        median = self.data[col].median()
        iqr = self.data[col].quantile(0.75) - self.data[col].quantile(0.25)
        text = f"Median {col} (IQR)"
        return self._to_dataframe(text, (median, iqr))

    def mean_and_ci(self, col: Union[str, list] = None) -> pd.DataFrame:
        """Mean and 95% confidence interval"""
        if col is None:
            return self.mean_and_ci(self.num)
        if not isinstance(col, str):
            return pd.concat([self.mean_and_ci(c) for c in col])

        col = col.lower()
        if not col in self.num:
            warnings.warn(f"{col} is not a numeric column.")
        mean = self.data[col].mean()
        ci = ci_(self.data[col])
        text = f"Mean {col} (95% CI)"
        return self._to_dataframe(text, (mean, ci))

    def counts(self, col: Union[str, list] = None) -> pd.DataFrame:
        """Return the counts of a column"""
        if col is None:
            return self.counts(self.cat)

        if not isinstance(col, str):
            return pd.concat([self.counts(c)
                              for c in col]).reset_index(drop=True)

        col = col.lower()
        if not col in self.cat:
            warnings.warn(f"{col} is not a categorical column.")
        # The counts, not normalized
        unnormed = self.data[col].value_counts(dropna=False)
        unnormed.name = self._columns[1]
        unnormed.index.name = self._columns[0]

        # The counts, normalized
        normed = unnormed / self.n
        normed.name = self._columns[2]
        normed.index.name = self._columns[0]

        top_row = pd.DataFrame([[col, np.nan, np.nan]], columns=self._columns)
        out = unnormed.to_frame().join(normed.to_frame()).reset_index()
        out[self._columns[0]] = out[self._columns[0]].astype(str)
        out = out.sort_values([self._columns[1], self._columns[0]])
        return pd.concat([top_row, out]).reset_index(drop=True)

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

    def prettify(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.set_index(df.columns[0])

        # To apply row by row
        def print_proper(x: pd.Series) -> str:
            if isinstance(x, str):
                return x
            # Confidence intervals
            if isinstance(x, tuple):
                return f"({x[0]:.02}, {x[1]:.02})"
            # Means and medians
            if x[self._columns[0]].str.match(r"Me(?:di)?an .* \(.+\)"):
                return f"({x[self._columns[2]]:.02f})"
            # Percentages
            return f"({100*x[self._columns[2]]:.02f}%)"

        out["parens"] = out.apply(print_proper, axis=1)
        out[""] = out[self._columns[1]] + " " + out["parens"]
        return out

    def analyze_categorical(self, as_str: bool = False) -> pd.DataFrame:
        out = self.counts(self.cat)
        if not as_str:
            return out
        return self.prettify(out)
