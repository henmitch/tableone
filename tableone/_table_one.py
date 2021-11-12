"""For the Table One class"""
import warnings
from typing import Tuple, Union
from ._helpers import ci as ci_

import pandas as pd


class TableOne():
    _columns = ["", "value", "paren"]

    def __init__(
        self,
        data: pd.DataFrame,
        categorical: list,
        numerical: list,
    ):
        self.data = data

        # Lowercase all column names
        def lower(x):
            return x.lower()

        self.data.columns = map(lower, self.data.columns)
        self.cat = map(lower, categorical)
        self.num = map(lower, numerical)
        missing = set(categorical) + set(numerical) - set(self.data.columns)
        if len(missing) > 0:
            raise ValueError(f"Missing columns: {missing}")
        self.data = self.data[self.cat + self.num]
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
        return pd.DataFrame([name, data[0], data[1]],
                            columns=TableOne._columns)

    def mean_and_sd(self, col: Union[str, list]) -> pd.DataFrame:
        """Return the mean and standard deviation of columns

        :param col: The column or columns to analyze.
        :type col: str or list

        :returns: A dataframe with the mean and standard deviation of the
            column.
        """
        if isinstance(col, str):
            return pd.concat([self.mean_and_sd(c) for c in col])

        mean = self.data[col].mean()
        sd = self.data[col].std()
        return self._to_dataframe(col, (mean, sd))

    def median_and_iqr(self, col: Union[str, list]) -> pd.DataFrame:
        """Return the median and interquartile range of columns

        :param col: The column or columns to analyze.
        :type col: str or list

        :returns: A dataframe with the median and interquartile range of the
            column.
        """
        if not isinstance(col, str):
            return pd.concat([self.median_and_iqr(c) for c in col])

        median = self.data[col].median()
        iqr = self.data[col].quantile(0.75) - self.data[col].quantile(0.25)
        return self._to_dataframe(col, (median, iqr))

    def counts(self, col: Union[str, list]) -> pd.DataFrame:
        """Return the counts of a column"""
        if not isinstance(col, str):
            return pd.concat([self.counts(c) for c in col])

        unnormed = self.data[col].value_counts(dropna=False)
        unnormed.name = self._columns[1]
        unnormed.index.name = self._columns[0]

        normed = unnormed / self.n
        normed.name = self._columns[2]
        normed.index.name = self._columns[0]

        top_row = pd.DataFrame(["col", "", ""], columns=self._columns)
        out = unnormed.to_frame().join(normed.to_frame())
        return pd.concat([top_row, out])

    def mean_and_ci(self, col: Union[str, list]) -> pd.DataFrame:
        if not isinstance(col, str):
            return pd.concat([self.mean_and_ci(c) for c in col])

        mean = self.data[col].mean()
        ci = ci_(self.data[col])
        return self._to_dataframe(col, (mean, ci))

    def count_na(self) -> pd.Series:
        """Return the number of missing values per column"""
        return self.data.isna().sum()

    def id_invalid(self) -> dict:
        """Identify the invalid values in the numeric columns"""
        out = dict()
        for column in self.num:
            try:
                self.data[column].to_numeric(errors="raise")
            except ValueError:
                coerced = self.data[column].to_numeric(errors="coerce")
                # Values that can't be coerced to numeric
                new_na = coerced[coerced.isna() & ~self.data.isna()].index
                out[column] = list(new_na)

        return out
