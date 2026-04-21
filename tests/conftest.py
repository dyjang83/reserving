import pytest
import pandas as pd
from reserving.triangle import Triangle


@pytest.fixture
def simple_triangle():
    """
    A small 3x3 upper-left triangle with known values.
    Used across all method tests for known-answer validation.

        lag:  1     2     3
    AY 2021: 1000  1100  1150
    AY 2022: 1200  1300   NaN
    AY 2023:  900   NaN   NaN
    """
    data = pd.DataFrame(
        {1: [1000, 1200, 900],
         2: [1100, 1300, None],
         3: [1150, None, None]},
        index=[2021, 2022, 2023]
    )
    return Triangle(data)


@pytest.fixture
def complete_triangle():
    """
    A fully developed 3x3 triangle — no missing values.
    Useful for testing methods that require complete data.

        lag:  1     2     3
    AY 2021: 1000  1100  1150
    AY 2022: 1200  1300  1380
    AY 2023:  900   990  1040
    """
    data = pd.DataFrame(
        {1: [1000, 1200, 900],
         2: [1100, 1300, 990],
         3: [1150, 1380, 1040]},
        index=[2021, 2022, 2023]
    )
    return Triangle(data)


@pytest.fixture
def long_df():
    """Long-format DataFrame matching simple_triangle."""
    return pd.DataFrame({
        "AY":   [2021, 2021, 2021, 2022, 2022, 2023],
        "lag":  [1,    2,    3,    1,    2,    1],
        "loss": [1000, 1100, 1150, 1200, 1300, 900],
    })


@pytest.fixture
def single_row_triangle():
    """Triangle with a single accident year."""
    data = pd.DataFrame(
        {1: [500], 2: [550], 3: [570]},
        index=[2021]
    )
    return Triangle(data)
