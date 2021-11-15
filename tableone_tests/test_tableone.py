"""Unit-testing the tableone module"""
import os
import unittest

import numpy as np
import pandas as pd

import tableone
from tableone._helpers import ci as ci_

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
        expect = pd.DataFrame(data=[["test", 1, 2]],
                              columns=["category", "value", "paren"])
        table = tableone.TableOne(pd.DataFrame(), [], [])

        pd.testing.assert_frame_equal(table._to_dataframe("test", (1, 2)),
                                      expect)
        pd.testing.assert_frame_equal(
            tableone.TableOne._to_dataframe("test", (1, 2)), expect)

    def test_mean_and_sd(self):
        """Test the mean and standard devation"""
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        table = tableone.TableOne(df, [], ["Col1", "Col2"])
        first_expect = pd.DataFrame(
            data=[["Mean col1 (SD)", 5.5,
                   np.std(df["Col1"], ddof=1)]],
            columns=["category", "value", "paren"])
        second_expect = pd.DataFrame(data=[["Mean col2 (SD)", 1.0, 0.0]],
                                     columns=["category", "value", "paren"])
        third_expect = pd.concat([first_expect, second_expect])

        pd.testing.assert_frame_equal(table.mean_and_sd("Col1"), first_expect)
        pd.testing.assert_frame_equal(table.mean_and_sd("Col2"), second_expect)
        pd.testing.assert_frame_equal(table.mean_and_sd(), third_expect)
        pd.testing.assert_frame_equal(table.mean_and_sd(["Col1", "Col2"]),
                                      third_expect)

    def test_mean_and_ci(self):
        """Test the mean and standard devation"""
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        table = tableone.TableOne(df, [], ["Col1", "Col2"])
        first_expect = pd.DataFrame(
            data=[["Mean col1 (95% CI)", 5.5,
                   ci_(df["Col1"])]],
            columns=["category", "value", "paren"])
        second_expect = pd.DataFrame(
            data=[["Mean col2 (95% CI)", 1.0,
                   ci_(df["Col2"])]],
            columns=["category", "value", "paren"])
        pd.testing.assert_frame_equal(table.mean_and_ci("Col1"), first_expect)
        pd.testing.assert_frame_equal(table.mean_and_ci("Col2"), second_expect)
        pd.testing.assert_frame_equal(table.mean_and_ci(),
                                      pd.concat([first_expect, second_expect]))

    def test_median_and_iqr(self):
        """Test the mean and standard devation"""
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        table = tableone.TableOne(df, [], ["Col1", "Col2"])
        first_expect = pd.DataFrame(data=[["Median col1 (IQR)", 5.5, 4.5]],
                                    columns=["category", "value", "paren"])
        second_expect = pd.DataFrame(data=[["Median col2 (IQR)", 1.0, 0.0]],
                                     columns=["category", "value", "paren"])
        pd.testing.assert_frame_equal(table.median_and_iqr("Col1"),
                                      first_expect)
        pd.testing.assert_frame_equal(table.median_and_iqr("Col2"),
                                      second_expect)
        pd.testing.assert_frame_equal(table.median_and_iqr(),
                                      pd.concat([first_expect, second_expect]))

    def test_counts(self):
        """Test the counts"""
        df = pd.read_csv(os.path.join(test_path, "test1.csv"))
        table = tableone.TableOne(df, ["Col1", "Col2"], [])

        counts_1_path = os.path.join(test_path, "counts1_expect.csv")
        counts_2_path = os.path.join(test_path, "counts2_expect.csv")
        first_expect = pd.read_csv(counts_1_path)
        second_expect = pd.read_csv(counts_2_path)
        third_expect = pd.concat([first_expect,
                                  second_expect]).reset_index(drop=True)

        with self.subTest("First column"):
            pd.testing.assert_frame_equal(table.counts("Col1"), first_expect,
            check_dtype=False, check_column_type=False)
        with self.subTest("Second column"):
            pd.testing.assert_frame_equal(table.counts("Col2"), second_expect)
        with self.subTest("All columns"):
            pd.testing.assert_frame_equal(table.counts(), third_expect)

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

    def test_count_na(self):
        """Test the NaN counting"""
        data = pd.read_csv(os.path.join(test_path, "test1.csv"))

        table = tableone.TableOne(data, ["Col1", "Col2"],
                                  ["Col1", "Col2", "Col3"])
        expect = pd.Series([0, 0, 1], index=["col1", "col2", "col3"])
        pd.testing.assert_series_equal(table.count_na(), expect)


if __name__ == '__main__':
    unittest.main()
