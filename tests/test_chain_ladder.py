import pytest
import numpy as np
import pandas as pd
from reserving.triangle import Triangle
from reserving.methods.chain_ladder import ChainLadder


class TestChainLadderFit:

    def test_returns_self(self, simple_triangle):
        cl = ChainLadder(simple_triangle)
        assert cl.fit() is cl

    def test_repr_unfitted(self, simple_triangle):
        cl = ChainLadder(simple_triangle)
        assert "not fitted" in repr(cl)

    def test_repr_fitted(self, simple_triangle):
        cl = ChainLadder(simple_triangle).fit()
        assert "fitted" in repr(cl)

    def test_rejects_non_triangle(self):
        with pytest.raises(TypeError):
            ChainLadder("not a triangle")

    def test_raises_before_fit(self, simple_triangle):
        cl = ChainLadder(simple_triangle)
        with pytest.raises(RuntimeError):
            cl.ultimates()
        with pytest.raises(RuntimeError):
            cl.ibnr()
        with pytest.raises(RuntimeError):
            cl.factors()


class TestChainLadderFactors:

    def test_factors_length(self, simple_triangle):
        cl = ChainLadder(simple_triangle).fit()
        # One factor per lag except the last
        assert len(cl.factors()) == simple_triangle.n_devs - 1

    def test_factors_known_values(self, simple_triangle):
        cl = ChainLadder(simple_triangle).fit()
        f = cl.factors()
        # f1 = (1100 + 1300) / (1000 + 1200) = 2400/2200
        assert abs(f.loc[1] - 2400 / 2200) < 1e-9
        # f2 = 1150 / 1100 (only one pair at lag 2→3)
        assert abs(f.loc[2] - 1150 / 1100) < 1e-9

    def test_factors_are_copy(self, simple_triangle):
        cl = ChainLadder(simple_triangle).fit()
        f = cl.factors()
        f.iloc[0] = 99999
        assert cl.factors().iloc[0] != 99999


class TestChainLadderUltimates:

    def test_ultimates_length(self, simple_triangle):
        cl = ChainLadder(simple_triangle).fit()
        assert len(cl.ultimates()) == simple_triangle.n_origins

    def test_ultimates_index(self, simple_triangle):
        cl = ChainLadder(simple_triangle).fit()
        assert list(cl.ultimates().index) == list(simple_triangle.origin_years)

    def test_ultimates_known_values(self, simple_triangle):
        cl = ChainLadder(simple_triangle).fit()
        u = cl.ultimates()
        f1 = 2400 / 2200
        f2 = 1150 / 1100
        # AY 2021: already at lag 3, no further development
        assert abs(u.loc[2021] - 1150.0) < 1e-9
        # AY 2022: 1300 * f2
        assert abs(u.loc[2022] - 1300 * f2) < 1e-9
        # AY 2023: 900 * f1 * f2
        assert abs(u.loc[2023] - 900 * f1 * f2) < 1e-9

    def test_complete_triangle_unchanged(self, complete_triangle):
        """For a complete triangle, ultimates equal the last column."""
        cl = ChainLadder(complete_triangle).fit()
        u = cl.ultimates()
        last_col = complete_triangle.data.iloc[:, -1]
        for ay in complete_triangle.origin_years:
            assert abs(u.loc[ay] - last_col.loc[ay]) < 1e-9

    def test_ultimates_geq_latest_diagonal(self, simple_triangle):
        """Ultimates must be >= latest observed (factors >= 1 for paid losses)."""
        cl = ChainLadder(simple_triangle).fit()
        diag = simple_triangle.latest_diagonal
        for ay in simple_triangle.origin_years:
            assert cl.ultimates().loc[ay] >= diag.loc[ay] - 1e-9


class TestChainLadderIBNR:

    def test_ibnr_nonnegative(self, simple_triangle):
        cl = ChainLadder(simple_triangle).fit()
        assert (cl.ibnr() >= -1e-9).all()

    def test_ibnr_equals_ultimate_minus_diagonal(self, simple_triangle):
        cl = ChainLadder(simple_triangle).fit()
        expected = cl.ultimates() - simple_triangle.latest_diagonal
        pd.testing.assert_series_equal(
            cl.ibnr().round(6), expected.rename("ibnr").round(6)
        )

    def test_ibnr_zero_for_fully_developed(self, simple_triangle):
        """AY 2021 is at the last lag — IBNR should be 0."""
        cl = ChainLadder(simple_triangle).fit()
        assert abs(cl.ibnr().loc[2021]) < 1e-9

    def test_total_ibnr(self, simple_triangle):
        cl = ChainLadder(simple_triangle).fit()
        assert abs(cl.total_ibnr() - cl.ibnr().sum()) < 1e-9


class TestChainLadderSummary:

    def test_summary_columns(self, simple_triangle):
        cl = ChainLadder(simple_triangle).fit()
        s = cl.summary(n_boot=100)
        for col in ["latest", "ultimate", "ibnr", "ci_lower", "ci_upper"]:
            assert col in s.columns

    def test_summary_shape(self, simple_triangle):
        cl = ChainLadder(simple_triangle).fit()
        s = cl.summary(n_boot=100)
        assert s.shape[0] == simple_triangle.n_origins

    def test_ci_bounds_ordered(self, simple_triangle):
        """ci_lower <= ultimate <= ci_upper for all accident years."""
        cl = ChainLadder(simple_triangle).fit()
        s = cl.summary(n_boot=200)
        assert (s["ci_lower"] <= s["ultimate"] + 1e-6).all()
        assert (s["ci_upper"] >= s["ultimate"] - 1e-6).all()

    def test_summary_index(self, simple_triangle):
        cl = ChainLadder(simple_triangle).fit()
        s = cl.summary(n_boot=100)
        assert list(s.index) == list(simple_triangle.origin_years)
