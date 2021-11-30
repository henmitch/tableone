"""Unit-testing the tableone module"""
import re
import ast
import os
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

import tableone
from tableone._helpers import (_category, _columns, _paren, _to_dataframe,
                               _value, booleanize)
from tableone._helpers import ci as ci_
from tableone._helpers import iqr, prettify

test_path = os.path.join(os.path.dirname(__file__), "test_data")


class TestTableOne(unittest.TestCase):
    """Unit-testing the tableone module"""
    def test_creation(self):
        """Test the creation of a TableOne object"""
        df = pd.DataFrame(columns=["Col1", "Col2"])
        table = tableone.TableOne(df, ["Col1"], ["Col2"])

        self.assertIsInstance(table, tableone.TableOne)
        self.assertEqual(table.cat, ["col1"])
        self.assertEqual(table.num, ["col2"])
        self.assertEqual(0, table.n)
        with self.assertRaises(ValueError):
            tableone.TableOne(df)

    def test_str(self):
        """Test the string representation"""
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        table = tableone.TableOne(df, [], ["Col1", "Col2"])
        expect = "TableOne(10 patients)"
        self.assertEqual(str(table), expect)

    def test_repr(self):
        """Test the string representation"""
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        table = tableone.TableOne(df, [], ["Col1", "Col2"])
        expect = "TableOne(10 patients)"
        self.assertEqual(repr(table), expect)

    def test_column_checking(self):
        """Test the column checking"""
        df = pd.DataFrame(columns=["Col1", "Col2", "Col3"])
        table = tableone.TableOne(df, ["Col1"], ["Col2"])

        # To check if extraneous columns get removed
        self.assertEqual(list(table.data.columns), ["col1", "col2"])
        # So that we can't add a column that is not in the data
        with self.assertRaises(ValueError):
            table = tableone.TableOne(df, ["Col1"], ["Col4"])

    def test__to_dataframe(self):
        """Test the to-dataframe conversion"""
        expect = pd.DataFrame(data=[["test", 1, 2]], columns=_columns)

        assert_frame_equal(_to_dataframe("test", (1, 2)), expect)
        assert_frame_equal(_to_dataframe("test", (1, 2)), expect)
        with self.assertWarns(UserWarning):
            _to_dataframe("test", (1, 2, 3))

    def test_mean_and_sd(self):
        """Test the mean and standard devation"""
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        table = tableone.TableOne(df, ["Col3"], ["Col1", "Col2"])
        first_expect = pd.DataFrame(
            data=[["Mean col1 (SD)", 5.5,
                   np.std(df["Col1"], ddof=1)]],
            columns=_columns)
        second_expect = pd.DataFrame(data=[["Mean col2 (SD)", 1.0, 0.0]],
                                     columns=_columns)
        third_expect = pd.concat([first_expect,
                                  second_expect]).reset_index(drop=True)

        assert_frame_equal(table.mean_and_sd("Col1"), first_expect)
        assert_frame_equal(table.mean_and_sd("Col2"), second_expect)
        assert_frame_equal(table.mean_and_sd(), third_expect)
        assert_frame_equal(table.mean_and_sd(["Col1", "Col2"]), third_expect)
        with self.assertWarns(UserWarning):
            table.mean_and_sd("Col3")

    def test_mean_and_sd_str(self):
        """Test the mean and standard devation"""
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        table = tableone.TableOne(df, ["Col3"], ["Col1", "Col2"])
        first_expect = pd.DataFrame(data=[[
            "Mean col1 (SD)", f"5.50 ({np.std(df['Col1'], ddof=1):.2f})"
        ]],
                                    columns=[_category, ""])
        second_expect = pd.DataFrame(data=[["Mean col2 (SD)", "1.00 (0.00)"]],
                                     columns=[_category, ""])
        third_expect = pd.concat([first_expect,
                                  second_expect]).reset_index(drop=True)

        assert_frame_equal(table.mean_and_sd("Col1", as_str=True),
                           first_expect)
        assert_frame_equal(table.mean_and_sd("Col2", as_str=True),
                           second_expect)
        assert_frame_equal(table.mean_and_sd(as_str=True), third_expect)
        assert_frame_equal(table.mean_and_sd(["Col1", "Col2"], as_str=True),
                           third_expect)

    def test_mean_and_ci(self):
        """Test the mean and standard devation"""
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        table = tableone.TableOne(df, ["Col3"], ["Col1", "Col2"])
        first_expect = pd.DataFrame(
            data=[["Mean col1 (95% CI)", 5.5,
                   ci_(df["Col1"])]],
            columns=_columns)
        second_expect = pd.DataFrame(
            data=[["Mean col2 (95% CI)", 1.0,
                   ci_(df["Col2"])]],
            columns=_columns)
        third_expect = pd.concat([first_expect,
                                  second_expect]).reset_index(drop=True)
        assert_frame_equal(table.mean_and_ci("Col1"), first_expect)
        assert_frame_equal(table.mean_and_ci("Col2"), second_expect)
        assert_frame_equal(table.mean_and_ci(), third_expect)
        assert_frame_equal(table.mean_and_ci(["Col1", "Col2"]), third_expect)
        with self.assertWarns(UserWarning):
            table.mean_and_ci("Col3")

    def test_mean_and_ci_str(self):
        """Test the mean and standard devation"""
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        table = tableone.TableOne(df, ["Col3"], ["Col1", "Col2"])
        col1_ci = ci_(df["Col1"])
        first_expect = pd.DataFrame(data=[[
            "Mean col1 (95% CI)", f"5.50 ({col1_ci[0]:.2f}, {col1_ci[1]:.2f})"
        ]],
                                    columns=[_category, ""])

        col2_ci = ci_(df["Col2"])
        second_expect = pd.DataFrame(data=[[
            "Mean col2 (95% CI)", f"1.00 ({col2_ci[0]:.2f}, {col2_ci[1]:.2f})"
        ]],
                                     columns=[_category, ""])
        third_expect = pd.concat([first_expect,
                                  second_expect]).reset_index(drop=True)

        assert_frame_equal(table.mean_and_ci("Col1", as_str=True),
                           first_expect)
        assert_frame_equal(table.mean_and_ci("Col2", as_str=True),
                           second_expect)
        assert_frame_equal(table.mean_and_ci(as_str=True), third_expect)
        assert_frame_equal(table.mean_and_ci(["Col1", "Col2"], as_str=True),
                           third_expect)
        with self.assertWarns(UserWarning):
            table.mean_and_ci("Col3")

    def test_median_and_iqr(self):
        """Test the mean and standard devation"""
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        table = tableone.TableOne(df, ["Col3"], ["Col1", "Col2"])
        first_expect = pd.DataFrame(data=[["Median col1 (IQR)", 5.5, 4.5]],
                                    columns=_columns)
        second_expect = pd.DataFrame(data=[["Median col2 (IQR)", 1.0, 0.0]],
                                     columns=_columns)
        third_expect = pd.concat([first_expect,
                                  second_expect]).reset_index(drop=True)
        assert_frame_equal(table.median_and_iqr("Col1"), first_expect)
        assert_frame_equal(table.median_and_iqr("Col2"), second_expect)
        assert_frame_equal(table.median_and_iqr(["Col1", "Col2"]),
                           third_expect)
        assert_frame_equal(table.median_and_iqr(), third_expect)
        with self.assertWarns(UserWarning):
            table.median_and_iqr("Col3")

    def test_counts(self):
        """Test the counts"""
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        table = tableone.TableOne(df, ["Col1", "Col2"], ["Col3"])

        counts_1_path = os.path.join(test_path, "counts1_expect.csv")
        counts_2_path = os.path.join(test_path, "counts2_expect.csv")
        first_expect = pd.read_csv(counts_1_path).fillna("")
        second_expect = pd.read_csv(counts_2_path).fillna("")
        third_expect = pd.concat([first_expect,
                                  second_expect]).reset_index(drop=True)

        assert_frame_equal(table.counts("Col1"),
                           first_expect,
                           check_dtype=False,
                           check_column_type=False)
        assert_frame_equal(table.counts("Col2"), second_expect)
        assert_frame_equal(table.counts(), third_expect)
        with self.assertWarns(UserWarning):
            table.counts("Col3")

    def test_id_invalid(self):
        """Test the data checking"""
        data = pd.read_csv(os.path.join(test_path, "test1.csv"))

        # Identifying invalid data
        table = tableone.TableOne(data, [], ["Col1", "Col2", "Col3"])
        invalid = table.id_invalid()
        self.assertEqual(invalid, {"col3": [0, 7]})

        # There should be no invalid data
        table = tableone.TableOne(data, [], ["Col1", "Col2"])
        invalid = table.id_invalid()
        self.assertEqual(invalid, {})

    def test_drop_invalid(self):
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        table = tableone.TableOne(df, [], ["Col1", "Col2", "Col3"])
        expect = df.copy()
        expect.columns = map(lambda x: x.lower(), expect.columns)
        expect.loc[[0, 4, 7], "col3"] = np.nan
        table.drop_invalid()
        assert_frame_equal(table.data, expect)

    def test_count_na(self):
        """Test the NaN counting"""
        data = pd.read_csv(os.path.join(test_path, "test1.csv"))

        table = tableone.TableOne(data, ["Col1", "Col2"],
                                  ["Col1", "Col2", "Col3"])
        expect = pd.Series([0, 0, 1], index=["col1", "col2", "col3"])
        pd.testing.assert_series_equal(table.count_na(), expect)

    def test_prettify(self):
        """Test the prettification"""
        data = pd.DataFrame({
            "category": [
                "String", "Tuple", "Mean value (paren)",
                "Median value (paren)", "Percent", "Empty"
            ],
            "Value":
                5 * [1.066] + [np.nan],
            "Paren": [
                "A string", (0.6634, 100.023), 0.0452, 0.0452, 0.04526, np.nan
            ]
        })
        expect = pd.DataFrame({
            "category":
                data["category"],
            "": [
                "1.07 (A string)", "1.07 (0.66, 100.02)", "1.07 (0.05)",
                "1.07 (0.05)", "1 (4.53%)", ""
            ]
        })
        assert_frame_equal(prettify(data), expect)

    def test_analyze_categorical_str(self):
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        expect = pd.read_csv(os.path.join(test_path,
                                          "categorical_analysis_expect.csv"),
                             header=0,
                             names=["category", ""])
        expect = expect.fillna("")
        table = tableone.TableOne(df, ["Col1", "Col2"], [])
        assert_frame_equal(table.analyze_categorical(as_str=True), expect)

    def test_analyze_categorical_no_str(self):
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        expect = pd.read_csv(os.path.join(test_path,
                                          "categorical_analysis_expect.csv"),
                             header=0,
                             names=["category", ""])
        expect = expect.join(expect[""].str.split(expand=True))
        expect = expect.rename(columns={0: "value", 1: "paren"})
        expect = expect[_columns]
        expect["value"] = expect["value"].astype(float)
        expect["paren"] = expect["paren"].str.extract(r"(\d+)").astype(
            float) / 100
        expect = expect.fillna("")
        table = tableone.TableOne(df, ["Col1", "Col2"], [])
        assert_frame_equal(table.analyze_categorical(), expect)

    def test_analyze_numeric_str(self):
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        expect = pd.DataFrame({
            "category": [
                "Mean col1 (SD)", "Mean col2 (SD)", "Mean col1 (95% CI)",
                "Mean col2 (95% CI)", "Median col1 (IQR)", "Median col2 (IQR)"
            ],
            "": [
                f"{df['Col1'].mean():.2f} ({df['Col1'].std():.2f})",
                f"{df['Col2'].mean():.2f} ({df['Col2'].std():.2f})",
                f"{df['Col1'].mean():.2f} ({ci_(df['Col1'])[0]:.2f}, "
                f"{ci_(df['Col1'])[1]:.2f})",
                f"{df['Col2'].mean():.2f} ({ci_(df['Col2'])[0]:.2f}, "
                f"{ci_(df['Col2'])[1]:.2f})",
                f"{df['Col1'].median():.2f} ({iqr(df['Col1']):.2f})",
                f"{df['Col2'].median():.2f} ({iqr(df['Col2']):.2f})"
            ]
        })
        table = tableone.TableOne(df, [], ["Col1", "Col2"])
        assert_frame_equal(table.analyze_numeric(as_str=True), expect)

    def test_numeric_invalid(self):
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        table = tableone.TableOne(df, [], ["Col1", "Col2", "Col3"])
        with self.assertRaises(ValueError):
            table.analyze_numeric(as_str=True)

    def test_analyze_numeric_no_str(self):
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        expect = pd.DataFrame({
            "category": [
                "Mean col1 (SD)", "Mean col2 (SD)", "Mean col1 (95% CI)",
                "Mean col2 (95% CI)", "Median col1 (IQR)", "Median col2 (IQR)"
            ],
            "value": [
                df['Col1'].mean(), df['Col2'].mean(), df['Col1'].mean(),
                df['Col2'].mean(), df['Col1'].median(), df['Col2'].median()
            ],
            "paren": [
                df['Col1'].std(), df['Col2'].std(),
                ci_(df['Col1']),
                ci_(df['Col2']),
                iqr(df['Col1']),
                iqr(df['Col2'])
            ]
        })
        table = tableone.TableOne(df, [], ["Col1", "Col2"])
        assert_frame_equal(table.analyze_numeric(), expect)

    def test_analyze_str(self):
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        expect_numeric = pd.DataFrame({
            "category": [
                "Mean col1 (SD)", "Mean col2 (SD)", "Mean col1 (95% CI)",
                "Mean col2 (95% CI)", "Median col1 (IQR)", "Median col2 (IQR)"
            ],
            "": [
                f"{df['Col1'].mean():.2f} ({df['Col1'].std():.2f})",
                f"{df['Col2'].mean():.2f} ({df['Col2'].std():.2f})",
                f"{df['Col1'].mean():.2f} ({ci_(df['Col1'])[0]:.2f}, "
                f"{ci_(df['Col1'])[1]:.2f})",
                f"{df['Col2'].mean():.2f} ({ci_(df['Col2'])[0]:.2f}, "
                f"{ci_(df['Col2'])[1]:.2f})",
                f"{df['Col1'].median():.2f} ({iqr(df['Col1']):.2f})",
                f"{df['Col2'].median():.2f} ({iqr(df['Col2']):.2f})"
            ]
        })
        expect_categorical = pd.read_csv(os.path.join(
            test_path, "categorical_analysis_expect.csv"),
                                         header=0,
                                         names=["category", ""])
        expect_categorical = expect_categorical.fillna("")
        expect = pd.concat([expect_categorical,
                            expect_numeric]).reset_index(drop=True)
        table = tableone.TableOne(df, ["Col1", "Col2"], ["Col1", "Col2"])
        assert_frame_equal(table.analyze(as_str=True), expect)

    def test_analyze_no_str(self):
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        expect_numeric = pd.DataFrame({
            "category": [
                "Mean col1 (SD)", "Mean col2 (SD)", "Mean col1 (95% CI)",
                "Mean col2 (95% CI)", "Median col1 (IQR)", "Median col2 (IQR)"
            ],
            "value": [
                df['Col1'].mean(), df['Col2'].mean(), df['Col1'].mean(),
                df['Col2'].mean(), df['Col1'].median(), df['Col2'].median()
            ],
            "paren": [
                df['Col1'].std(), df['Col2'].std(),
                ci_(df['Col1']),
                ci_(df['Col2']),
                iqr(df['Col1']),
                iqr(df['Col2'])
            ]
        })
        expect_categorical = pd.read_csv(os.path.join(
            test_path, "categorical_analysis_expect.csv"),
                                         header=0,
                                         names=["category", ""])
        expect_categorical = expect_categorical.fillna("")
        expect_categorical = expect_categorical.join(
            expect_categorical[""].str.split(expand=True))
        expect_categorical = expect_categorical.rename(columns={
            0: "value",
            1: "paren"
        })
        expect_categorical = expect_categorical[_columns]
        expect_categorical["value"] = expect_categorical["value"].astype(float)
        expect_categorical["paren"] = expect_categorical["paren"].str.extract(
            "(\d+)").astype(float) / 100
        expect = pd.concat([expect_categorical,
                            expect_numeric]).reset_index(drop=True)
        expect = expect.fillna("")

        table = tableone.TableOne(df, ["Col1", "Col2"], ["Col1", "Col2"])
        assert_frame_equal(table.analyze(), expect)

    def test_invalid_groupings(self):
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        with self.assertRaises(ValueError):
            tableone.TableOne(df, ["Col1", "Col2"], ["Col1", "Col2"],
                              groupings=["Col3"])

    def test__split_groupings_one(self):
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        table = tableone.TableOne(df, ["Col1", "Col2"], ["Col1", "Col2"],
                                  groupings="Col1")

        def create_expect(i):
            return pd.DataFrame({"col1": i, "col2": 1}, index=[0])

        expect = {f"col1 = {i}": create_expect(i) for i in df['Col1'].unique()}
        real = table._split_groupings()

        for idx in expect:
            with self.subTest(idx=idx):
                assert_frame_equal(expect[idx], real[idx])

    def test__split_groupings_two(self):
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        table = tableone.TableOne(df, ["Col1", "Col2"], ["Col1", "Col2"],
                                  groupings=["Col1", "Col2"])

        def create_expect(i):
            return pd.DataFrame({"col1": i, "col2": 1}, index=[0])

        expect = {f"col1 = {i}": create_expect(i) for i in df['Col1'].unique()}
        expect |= {"col2 = 1": table.data}
        real = table._split_groupings()

        for idx in expect:
            with self.subTest(idx=idx):
                assert_frame_equal(expect[idx], real[idx])

    def test_mean_and_sd_groupings_str(self):
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        expect = pd.read_csv(
            os.path.join(test_path, "numerical_expect_groupings_str.csv"))
        expect = expect.rename(columns={"Unnamed: 1": ""})
        expect = expect[expect[_category].str.match(r"Mean col\d \(SD\)")]

        table = tableone.TableOne(df, ["Col1", "Col2"], ["Col1", "Col2"],
                                  groupings=["Col1", "Col2"])
        assert_frame_equal(table.mean_and_sd(as_str=True), expect)

        expect_2 = expect.drop(columns="col2 = 1 (n = 10)")
        table_2 = tableone.TableOne(df, ["Col1", "Col2"], ["Col1", "Col2"],
                                    groupings=["Col1"])
        assert_frame_equal(table_2.mean_and_sd(as_str=True), expect_2)

    def test_mean_and_ci_groupings_str(self):
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        expect = pd.read_csv(
            os.path.join(test_path, "numerical_expect_groupings_str.csv"))
        expect = expect.rename(columns={"Unnamed: 1": ""})
        expect = expect[expect[_category].str.match(r"Mean col\d \(95% CI\)")]
        expect = expect.reset_index(drop=True)

        table = tableone.TableOne(df, ["Col1", "Col2"], ["Col1", "Col2"],
                                  groupings=["Col1", "Col2"])
        assert_frame_equal(table.mean_and_ci(as_str=True), expect)

        expect_2 = expect.drop(columns="col2 = 1 (n = 10)")
        table_2 = tableone.TableOne(df, ["Col1", "Col2"], ["Col1", "Col2"],
                                    groupings=["Col1"])
        assert_frame_equal(table_2.mean_and_ci(as_str=True), expect_2)

    def test_median_and_iqr_groupings_str(self):
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        expect = pd.read_csv(
            os.path.join(test_path, "numerical_expect_groupings_str.csv"))
        expect = expect.rename(columns={"Unnamed: 1": ""})
        expect = expect[expect[_category].str.match(r"Median")]
        expect = expect.reset_index(drop=True)

        table = tableone.TableOne(df, ["Col1", "Col2"], ["Col1", "Col2"],
                                  groupings=["Col1", "Col2"])
        assert_frame_equal(table.median_and_iqr(as_str=True), expect)

        expect_2 = expect.drop(columns="col2 = 1 (n = 10)")
        table_2 = tableone.TableOne(df, ["Col1", "Col2"], ["Col1", "Col2"],
                                    groupings=["Col1"])
        assert_frame_equal(table_2.median_and_iqr(as_str=True), expect_2)

    def test_mean_and_sd_groupings_no_str(self):
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        expect = pd.read_csv(
            os.path.join(test_path, "numerical_expect_groupings_no_str.csv"))
        expect = expect.rename(columns={"Unnamed: 1": ""})
        expect = expect[expect[_category].str.match(r"Mean col\d \(SD\)")]
        expect = expect.reset_index(drop=True)
        expect = expect.replace({"None": np.nan})
        for col in expect.columns:
            expect[col] = expect[col].astype(float, errors="ignore")

        table = tableone.TableOne(df, ["Col1", "Col2"], ["Col1", "Col2"],
                                  groupings=["Col1", "Col2"])
        assert_frame_equal(table.mean_and_sd(), expect)

        expect_2 = expect.drop(
            columns=["value (col2 = 1)", "paren (col2 = 1)"])
        table_2 = tableone.TableOne(df, ["Col1", "Col2"], ["Col1", "Col2"],
                                    groupings=["Col1"])
        assert_frame_equal(table_2.mean_and_sd(), expect_2)

    def test_mean_and_ci_groupings_no_str(self):
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        expect = pd.read_csv(
            os.path.join(test_path, "numerical_expect_groupings_no_str.csv"),
            converters={
                "paren": ast.literal_eval,
                "paren (col2 = 1)": ast.literal_eval
            }
            | {f"paren (col1 = {i})": ast.literal_eval
               for i in range(1, 11)})
        expect = expect.rename(columns={"Unnamed: 1": ""})
        expect = expect[expect[_category].str.match(r"Mean col\d \(95% CI\)")]
        expect = expect.reset_index(drop=True)
        for col in expect.columns:
            expect[col] = expect[col].astype(float, errors="ignore")
        # expect = expect.replace("(nan, nan)", (np.nan, np.nan))

        table = tableone.TableOne(df, ["Col1", "Col2"], ["Col1", "Col2"],
                                  groupings=["Col1", "Col2"])

        to_explode = list(c for c in expect.columns if re.match("paren", c))
        assert_frame_equal(table.mean_and_ci().explode(to_explode),
                           expect.explode(to_explode))

        expect_2 = expect.drop(
            columns=["value (col2 = 1)", "paren (col2 = 1)"])
        to_explode_2 = list(c for c in expect_2.columns
                            if re.match("paren", c))
        table_2 = tableone.TableOne(df, ["Col1", "Col2"], ["Col1", "Col2"],
                                    groupings=["Col1"])
        assert_frame_equal(table_2.mean_and_ci().explode(to_explode_2),
                           expect_2.explode(to_explode_2))

    def test_median_and_iqr_groupings_no_str(self):
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        expect = pd.read_csv(
            os.path.join(test_path, "numerical_expect_groupings_no_str.csv"))
        expect = expect.rename(columns={"Unnamed: 1": ""})
        expect = expect[expect[_category].str.match(r"Median")]
        expect = expect.reset_index(drop=True)
        for col in expect.columns:
            expect[col] = expect[col].astype(float, errors="ignore")

        table = tableone.TableOne(df, ["Col1", "Col2"], ["Col1", "Col2"],
                                  groupings=["Col1", "Col2"])
        assert_frame_equal(table.median_and_iqr(), expect)

        expect_2 = expect.drop(
            columns=["value (col2 = 1)", "paren (col2 = 1)"])
        table_2 = tableone.TableOne(df, ["Col1", "Col2"], ["Col1", "Col2"],
                                    groupings=["Col1"])
        assert_frame_equal(table_2.median_and_iqr(), expect_2)

    def test_counts_groupings_str(self):
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        expect = pd.read_csv(
            os.path.join(test_path, "counts_expect_groupings_str.csv"))
        expect = expect.rename(columns={"Unnamed: 1": ""}).fillna("")

        table = tableone.TableOne(df, ["Col1", "Col2"],
                                  groupings=["Col1", "Col2"])
        assert_frame_equal(table.counts(as_str=True).fillna(""), expect)

    def test_counts_groupings_no_str(self):
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        expect = pd.read_csv(os.path.join(
            test_path, "counts_expect_groupings_no_str.csv"),
                             index_col=False)
        expect = expect.rename(columns={"Unnamed: 1": ""}).fillna("")

        table = tableone.TableOne(df, ["Col1", "Col2"],
                                  groupings=["Col1", "Col2"])
        assert_frame_equal(table.counts().fillna(""), expect)

    def test_booleanize_no_fillna(self):
        df = pd.read_csv(os.path.join(test_path, "booleanize_expect.csv"))
        expect = pd.Series([False, True, np.nan, True])
        for col in df.columns:
            with self.subTest(col=col):
                expect.name = col
                assert_series_equal(booleanize(df[col]), expect)

        expect2 = pd.Series([False, True, False, True])
        assert_series_equal(booleanize(expect2), expect2)

    def test_booleanize__fillna(self):
        df = pd.read_csv(os.path.join(test_path, "booleanize_expect.csv"))
        expect = pd.Series([False, True, False, True])
        for col in df.columns:
            with self.subTest(col=col):
                expect.name = col
                assert_series_equal(booleanize(df[col], fillna=False), expect)

    def test_boolean_str(self):
        df = pd.read_csv(os.path.join(test_path, "boolean_input.csv"))
        expect = pd.read_csv(os.path.join(test_path, "boolean_expect_str.csv"))
        expect = expect.rename(columns={"Unnamed: 1": ""})
        table = tableone.TableOne(df,
                                  boolean=["col1", "col2"],
                                  groupings=["col2"])
        assert_frame_equal(table.analyze_boolean(as_str=True), expect)

    def test_boolean_no_str(self):
        df = pd.read_csv(os.path.join(test_path, "boolean_input.csv"))
        expect = pd.read_csv(
            os.path.join(test_path, "boolean_expect_no_str.csv"))
        expect = expect.rename(columns={"Unnamed: 1": ""}).astype(object)
        table = tableone.TableOne(df,
                                  boolean=["col1", "col2"],
                                  groupings=["col2"])
        assert_frame_equal(table.analyze_boolean(), expect)


if __name__ == '__main__':
    unittest.main()
