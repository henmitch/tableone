from typing import Tuple
import pandas as pd
import scipy.stats
import numpy as np


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
