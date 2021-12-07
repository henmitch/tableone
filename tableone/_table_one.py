"""For the Table One class"""
import warnings
from typing import Callable, Dict, Tuple, Union

import numpy as np
import pandas as pd

from ._helpers import _category, _paren, _value, categorical_calculation
from ._helpers import ci as ci_
from ._helpers import iqr, numerical_calculation, prettify


class TableOne:
    """A class to hold and analyze data for a Table One

    :param data: The data to analyze.
    :type data: pandas.DataFrame
    :param categorical: The categorical columns to analyze.
    :type categorical: list
    :param numerical: The numerical columns to analyze.
    :type numerical: list
    :param groupings: The columns to group by.
    :type groupings: list
    """
    def __init__(self,
                 data: pd.DataFrame,
                 categorical: list,
                 numerical: list,
                 groupings: Union[list, str] = None):
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

        if isinstance(groupings, str):
            groupings = [groupings]
        self.groupings = [] if groupings is None else list(
            map(lower, groupings))
        if missing_groupings := (set(self.groupings) - set(self.data.columns)):
            raise ValueError("Groupings must be columns in data."
                             f" Missing {missing_groupings}")

    def __repr__(self):
        return f"TableOne({self.n} patients)"

    def __str__(self) -> str:
        return f"TableOne({self.n} patients)"

    def mean_and_sd(self,
                    col: Union[str, list] = None,
                    as_str: bool = False) -> pd.DataFrame:
        """Return the mean and standard deviation of columns

        :param col: The column or columns to analyze.
        :type col: str or list

        :returns: A dataframe with the mean and standard deviation of the
            column.
        """
        def mean(c: pd.Series) -> float:
            """Return the mean of a column"""
            return c.mean()

        mean.__name__ = "Mean"

        def sd(c: pd.Series) -> float:
            """Return the standard deviation of a column"""
            return c.std()

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
        def median(c: pd.Series) -> float:
            """Return the median of a column"""
            return c.median()

        median.__name__ = "Median"

        def _iqr(c: pd.Series) -> float:
            """Return the interquartile range of a column"""
            return iqr(c)

        _iqr.__name__ = "IQR"

        return self._num_calc(col, median, _iqr, as_str=as_str)

    def mean_and_ci(self,
                    col: Union[str, list] = None,
                    as_str: bool = False) -> pd.DataFrame:
        """Mean and 95% confidence interval"""

        # Just for naming purposes
        def _mean(c: pd.Series) -> float:
            """Return the mean of a column"""
            return c.mean()

        _mean.__name__ = "Mean"

        def _ci(c: pd.Series) -> Tuple[float, float]:
            """Return the 95% confidence interval of a column"""
            return ci_(c)

        _ci.__name__ = "95% CI"

        return self._num_calc(col, _mean, _ci, as_str=as_str)

    def _num_calc(self,
                  col_name: Union[str, list],
                  val_func: Callable,
                  spread_func: Callable,
                  text: str = None,
                  as_str: bool = False,
                  name: str = "",
                  split: bool = True) -> pd.DataFrame:
        """A generic function to calculate center and spread

        :param col_name: The column or columns to analyze.
        :type col_name: str or list
        :param val_func: The function to calculate the value.
        :type val_func: Callable
        :param spread_func: The function to calculate the spread.
        :type spread_func: Callable
        :param text: The text to use for the resultant row.
        :type text: str
        :param as_str: Whether to return the result as a dataframe of strings.
        :type as_str: bool
        :param name: The name of the resultant column.
        :type name: str
        :param split: Whether to split the column based on groupings.
        :type split: bool

        :return: A dataframe with the center and spread of the data
        :rtype: pd.DataFrame
        """
        params = {
            "val_func": val_func,
            "spread_func": spread_func,
            "text": text
        }
        data = self.data.copy()

        # The groupings to split into
        split_groupings = self._split_groupings()
        names = [f"All patients (n = {self.n})"] + [
            f"{idx} (n = {len(group)})"
            for idx, group in split_groupings.items()
        ]

        # If we're only analyzing one column, we need it lowercase
        if isinstance(col_name, str):
            col_name = col_name.lower()
            if not col_name in self.num:
                warnings.warn(f"{col_name} is not a numeric column.")
                data[col_name] = pd.to_numeric(col_name, errors="coerce")

        # We can look at all columns together, since they'll be uniquely
        # indexed
        if isinstance(col_name, list):
            col_name = list(map(str.lower, col_name))
        # If unspecified, just do all of them.
        if col_name is None:
            col_name = self.num

        # TODO: Refactor this
        if (not self.groupings) or (not split):
            out = numerical_calculation(data[col_name], **params, name=name)
            if as_str:
                out = prettify(out, name=names[0])
            return out

        out = numerical_calculation(data[col_name], **params, name=name)
        for idx, group in split_groupings.items():
            calc = numerical_calculation(group[col_name],
                                         **params,
                                         name=f"{idx} (n = {len(group)})")
            calc.columns = [_category
                            ] + [f"{c} ({idx})" for c in [_value, _paren]]
            out = out.merge(calc, how="outer", suffixes=["", " " + idx])

        if as_str:
            out = prettify(out, name=names)

        return out

    def counts(self,
               col_name: Union[str, list] = None,
               as_str: bool = False,
               name: str = "",
               split: bool = True) -> pd.DataFrame:
        """Return the counts of a column"""
        data = self.data.copy()
        if col_name is None:
            return self.counts(self.cat, as_str=as_str)

        split_groupings = self._split_groupings()
        names = [f"All patients (n = {self.n})"] + [
            f"{idx} (n = {len(group)})"
            for idx, group in split_groupings.items()
        ]
        if not isinstance(col_name, str):
            out = pd.concat([
                self.counts(c, as_str=False, name=name, split=split)
                for c in col_name
            ]).fillna(0).reset_index(drop=True)
            if as_str:
                out = prettify(out, name=names)
            return out

        col_name = col_name.lower()
        if not col_name in self.cat:
            warnings.warn(f"{col_name} is not a categorical column.")

        if (not self.groupings) or (not split):
            out = categorical_calculation(data[col_name], name=name)
            if as_str:
                out = prettify(out, name=names[0])
            return out

        out = categorical_calculation(data[col_name], name=name)
        for idx, group in split_groupings.items():
            calc = categorical_calculation(group[col_name],
                                           name=f"{idx} (n = {len(group)})")
            calc.columns = [_category
                            ] + [f"{c} ({idx})" for c in [_value, _paren]]

            out = out.merge(calc,
                            on=_category,
                            how="outer",
                            suffixes=["", " " + idx])

        out = out.fillna(0)

        if as_str:
            out = prettify(out, name=names)

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

    def _split_groupings(self) -> Dict[str, pd.DataFrame]:
        """Split the data into the provided groups"""
        out = dict()
        for col in self.groupings:
            grpby = self.data.groupby(col)
            out |= {
                f"{col} = {idx}": grp.reset_index(drop=True)
                for idx, grp in grpby
            }
        return out
