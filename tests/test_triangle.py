import pytest
import pandas as pd
import numpy as np
from reserving.triangle import Triangle


class TestTriangleConstruction:

    def test_basic_construction(self, simple_triangle):
        assert isinstance(simple_triangle, Triangle)

    def test_shape(self, simple_triangle):
        assert simple_triangle.shape == (3, 3)

    def test_index_names(self, simple_triangle):
        assert simple_triangle.data.index.name == "accident_year"
        assert simple_triangle.data.columns.name == "dev_lag"

    def test_data_is_float(self, simple_triangle):
        assert simple_triangle.data.dtypes.unique()[0] == float

    def test_data_is_copy(self, simple_triangle):
        """Modifying returned data should not affect internal state."""
        d = simple_triangle.data
        d.iloc[0, 0] = 99999
        assert simple_triangle.data.iloc[0, 0] != 99999

    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError):
            Triangle([[1, 2], [3, 4]])

    def test_rejects_empty_dataframe(self):
        with pytest.raises(ValueError):
            Triangle(pd.DataFrame())

    def test_from_dataframe(self, long_df):
        tri = Triangle.from_dataframe(long_df, origin="AY", dev="lag", values="loss")
        assert tri.shape == (3, 3)
        assert tri.data.loc[2021, 1] == 1000.0
        assert tri.data.loc[2022, 2] == 1300.0

    def test_from_dataframe_missing_column(self, long_df):
        with pytest.raises(ValueError):
            Triangle.from_dataframe(long_df, origin="AY", dev="lag", values="nonexistent")

    def test_from_dataframe_matches_direct(self, simple_triangle, long_df):
        tri = Triangle.from_dataframe(long_df, origin="AY", dev="lag", values="loss")
        pd.testing.assert_frame_equal(
            tri.data.fillna(0),
            simple_triangle.data.fillna(0)
        )


class TestTriangleProperties:

    def test_n_origins(self, simple_triangle):
        assert simple_triangle.n_origins == 3

    def test_n_devs(self, simple_triangle):
        assert simple_triangle.n_devs == 3

    def test_origin_years(self, simple_triangle):
        assert list(simple_triangle.origin_years) == [2021, 2022, 2023]

    def test_dev_lags(self, simple_triangle):
        assert list(simple_triangle.dev_lags) == [1, 2, 3]

    def test_is_complete_false(self, simple_triangle):
        assert simple_triangle.is_complete is False

    def test_is_complete_true(self, complete_triangle):
        assert complete_triangle.is_complete is True

    def test_latest_diagonal(self, simple_triangle):
        diag = simple_triangle.latest_diagonal
        assert diag.loc[2021] == 1150.0
        assert diag.loc[2022] == 1300.0
        assert diag.loc[2023] == 900.0

    def test_latest_dev_lag(self, simple_triangle):
        lags = simple_triangle.latest_dev_lag
        assert lags.loc[2021] == 3
        assert lags.loc[2022] == 2
        assert lags.loc[2023] == 1

    def test_len(self, simple_triangle):
        assert len(simple_triangle) == 3


class TestTriangleMethods:

    def test_link_ratios_shape(self, simple_triangle):
        ratios = simple_triangle.link_ratios()
        assert ratios.shape == simple_triangle.shape

    def test_link_ratios_last_col_nan(self, simple_triangle):
        ratios = simple_triangle.link_ratios()
        assert ratios.iloc[:, -1].isna().all()

    def test_link_ratios_known_values(self, simple_triangle):
        ratios = simple_triangle.link_ratios()
        assert abs(ratios.loc[2021, 1] - 1100 / 1000) < 1e-9
        assert abs(ratios.loc[2021, 2] - 1150 / 1100) < 1e-9

    def test_volume_weighted_factors_length(self, simple_triangle):
        factors = simple_triangle.volume_weighted_factors()
        assert len(factors) == simple_triangle.n_devs - 1

    def test_volume_weighted_factors_known_values(self, simple_triangle):
        factors = simple_triangle.volume_weighted_factors()
        expected_f1 = (1100 + 1300) / (1000 + 1200)
        assert abs(factors.loc[1] - expected_f1) < 1e-9

    def test_to_incremental_first_col_unchanged(self, simple_triangle):
        inc = simple_triangle.to_incremental()
        assert inc.data.iloc[0, 0] == simple_triangle.data.iloc[0, 0]

    def test_to_incremental_known_values(self, simple_triangle):
        inc = simple_triangle.to_incremental()
        assert inc.data.loc[2021, 2] == 100.0
        assert inc.data.loc[2021, 3] == 50.0

    def test_to_incremental_returns_triangle(self, simple_triangle):
        inc = simple_triangle.to_incremental()
        assert isinstance(inc, Triangle)

    def test_summary_shape(self, simple_triangle):
        s = simple_triangle.summary()
        assert s.shape[0] == simple_triangle.n_devs
        assert "n_obs" in s.columns

    def test_summary_counts(self, simple_triangle):
        s = simple_triangle.summary()
        assert s.loc[1, "n_obs"] == 3
        assert s.loc[2, "n_obs"] == 2
        assert s.loc[3, "n_obs"] == 1


class TestTriangleDunder:

    def test_repr(self, simple_triangle):
        r = repr(simple_triangle)
        assert "Triangle" in r
        assert "origins=3" in r
        assert "complete=False" in r

    def test_str(self, simple_triangle):
        s = str(simple_triangle)
        assert "2021" in s
        assert "1000" in s

    def test_getitem(self, simple_triangle):
        col = simple_triangle[1]
        assert col.loc[2021] == 1000.0

    def test_single_row(self, single_row_triangle):
        assert single_row_triangle.n_origins == 1
        assert single_row_triangle.is_complete is True
