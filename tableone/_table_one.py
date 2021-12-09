"""For the Table One class"""
import warnings
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from ._helpers import (_category, _paren, _value, categorical_calculation,
                       chi_square)
from ._helpers import ci as ci_
from ._helpers import iqr, median_test, numerical_calculation, prettify, ttest


class TableOne:
    """A class to hold and analyze data for a Table One

    :param data: The data to analyze.
    :type data: pandas.DataFrame
    :param categorical: The categorical columns to analyze.
    :type categorical: list
    :param numerical: The numerical columns to analyze.
    :type numerical: list
    :param groupings: The columns to group by.
    :type groupings: list or str
    :param compare: The groupings to perform comparison tests on.
    :type compare: list or str
    """
    def __init__(self,
                 data: pd.DataFrame,
                 categorical: list,
                 numerical: list,
                 groupings: Union[List[str], str] = None,
                 comparisons: Union[List[str], str] = None):
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

        if invalid := self.id_invalid():
            warnings.warn("Dropping invalid values found in numeric columns: "
                          f"{invalid}")
        self.drop_invalid()

        for col in self.num:
            self.data[col] = pd.to_numeric(self.data[col])

        if isinstance(groupings, str):
            groupings = [groupings]
        self.groupings = [] if groupings is None else list(
            map(lower, groupings))
        if missing_groupings := (set(self.groupings) - set(self.data.columns)):
            raise ValueError("Groupings must be columns in data."
                             f" Missing {missing_groupings}")

        if isinstance(comparisons, str):
            comparisons = [comparisons]
        self.comparisons = [] if comparisons is None else list(
            map(lower, comparisons))
        if missing_compare := (set(self.comparisons) - set(self.groupings)):
            raise ValueError("Comparison groupings must be groupings."
                             f" Missing {missing_compare}")
        if any(self.data[c].nunique() > 2 for c in self.comparisons):
            raise ValueError("Comparison groups must have at most 2 values.")

    def __repr__(self):
        return f"TableOne({self.n} patients)"

    def __str__(self) -> str:
        return f"TableOne({self.n} patients)"

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

    def _calc(self,
              col_name: Union[str, list],
              func: Callable,
              as_str: bool = False,
              name: str = "",
              split: bool = True,
              test: Callable[[pd.Series, pd.Series], float] = None,
              **params) -> pd.DataFrame:
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
        data = self.data.copy()

        # The groupings to split into
        split_groupings = self._split_groupings()
        names = [f"All patients (n = {self.n})"] + [
            f"{idx} (n = {len(group)})"
            for idx, group in split_groupings.items()
        ]

        # If we're only analyzing one column, we need it lowercase
        if not isinstance(col_name, str):
            out = pd.concat([
                self._calc(c,
                           func,
                           as_str=False,
                           name=name,
                           split=split,
                           test=test,
                           **params) for c in col_name
            ]).reset_index(drop=True)
            if as_str:
                out = prettify(out, name=names)
            return out

        col_name = col_name.lower()

        if (not self.groupings) or (not split):
            # Just doing all the data
            out = func(data[col_name], name=name, **params)
            if as_str:
                out = prettify(out, name=names[0])
            return out

        # Analyizing each group separately
        out = func(data[col_name], name=name, **params)
        for idx, group in split_groupings.items():
            calc = func(group[col_name],
                        name=f"{idx} (n = {len(group)})",
                        **params)
            calc.columns = [_category
                            ] + [f"{c} ({idx})" for c in [_value, _paren]]
            out = out.merge(calc, how="outer", suffixes=["", " " + idx])

        # Comparison tests
        if self.comparisons and test is not None:
            for comp in self.comparisons:
                row_idx = out.index[0]
                out.loc[row_idx, f"p ({comp})"] = test(data[col_name],
                                                       data[comp])

        if as_str:
            out = prettify(out, name=names)

        return out

    def _num_calc(self,
                  col_name: Union[str, list],
                  val_func: Callable,
                  spread_func: Callable,
                  text: str = None,
                  as_str: bool = False,
                  name: str = "",
                  split: bool = True,
                  test: Callable = None) -> pd.DataFrame:
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
        if col_name is None:
            col_name = self.num

        out = self._calc(col_name,
                         func=numerical_calculation,
                         as_str=as_str,
                         name=name,
                         val_func=val_func,
                         spread_func=spread_func,
                         text=text,
                         split=split,
                         test=test)

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
        for column, idxes in invalid.items():
            self.data.loc[idxes, column] = np.nan
            self.data[column] = self.data[column].astype(float)

    def analyze_numeric(self, as_str: bool = False) -> pd.DataFrame:
        out = pd.concat([
            self.mean_and_sd(self.num, as_str=as_str),
            self.mean_and_ci(self.num, as_str=as_str),
            self.median_and_iqr(self.num, as_str=as_str)
        ]).reset_index(drop=True)
        return out

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

        return self._num_calc(col, mean, sd, as_str=as_str, test=ttest)

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

        return self._num_calc(col,
                              median,
                              _iqr,
                              as_str=as_str,
                              test=median_test)

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

        return self._num_calc(col, _mean, _ci, as_str=as_str, test=ttest)

    def analyze_categorical(self, as_str: bool = False) -> pd.DataFrame:
        out = self.counts(self.cat, as_str=as_str).reset_index(drop=True)
        return out

    def counts(self,
               col_name: Union[str, list] = None,
               as_str: bool = False,
               name: str = "",
               split: bool = True) -> pd.DataFrame:
        """Return the counts of a column"""
        if col_name is None:
            col_name = self.cat

        out = self._calc(col_name,
                         func=categorical_calculation,
                         as_str=as_str,
                         name=name,
                         split=split,
                         test=chi_square)

        return out

    def analyze(self, as_str: bool = False) -> pd.DataFrame:
        out = pd.concat([
            self.analyze_categorical(as_str=as_str),
            self.analyze_numeric(as_str=as_str)
        ]).reset_index(drop=True)
        return out
