import pandas as pd
import numpy as np


class Triangle:
    """
    Core data structure for a loss development triangle.

    A triangle represents cumulative losses (paid or incurred) organized
    by accident year (rows) and development lag (columns). It is the
    input to all reserving methods in this library.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame in triangle format — accident years as the index,
        development lags as columns, loss values as cells.
        Missing cells (future periods) should be NaN.

    Examples
    --------
    >>> import pandas as pd
    >>> from reserving import Triangle
    >>> data = pd.DataFrame(
    ...     {1: [1000, 1200, 900], 2: [1100, 1300, None], 3: [1150, None, None]},
    ...     index=[2021, 2022, 2023]
    ... )
    >>> tri = Triangle(data)
    >>> tri.shape
    (3, 3)
    """

    def __init__(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data must be a DataFrame, got {type(data).__name__}")
        if data.empty:
            raise ValueError("data must not be empty")
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("data must have at least one row and one column")

        self._data = data.copy().astype(float)
        self._data.index.name = "accident_year"
        self._data.columns.name = "dev_lag"

    # ------------------------------------------------------------------ #
    # Class methods — alternative constructors                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        origin: str,
        dev: str,
        values: str,
    ) -> "Triangle":
        """
        Construct a Triangle from a long-format DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Long-format data with one row per origin/development observation.
        origin : str
            Column name for accident year (rows of the triangle).
        dev : str
            Column name for development lag (columns of the triangle).
        values : str
            Column name for the loss values.

        Returns
        -------
        Triangle

        Examples
        --------
        >>> tri = Triangle.from_dataframe(df, origin="AccidentYear",
        ...                               dev="DevelopmentLag",
        ...                               values="CumPaidLoss")
        """
        required = {origin, dev, values}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

        pivot = df.pivot_table(
            index=origin, columns=dev, values=values, aggfunc="first"
        )
        pivot.index.name = "accident_year"
        pivot.columns.name = "dev_lag"
        return cls(pivot)

    @classmethod
    def from_csv(
        cls,
        path: str,
        origin: str,
        dev: str,
        values: str,
        **kwargs,
    ) -> "Triangle":
        """
        Construct a Triangle from a CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV file.
        origin : str
            Column name for accident year.
        dev : str
            Column name for development lag.
        values : str
            Column name for loss values.
        **kwargs
            Additional keyword arguments passed to pd.read_csv.

        Returns
        -------
        Triangle
        """
        df = pd.read_csv(path, **kwargs)
        return cls.from_dataframe(df, origin=origin, dev=dev, values=values)

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def data(self) -> pd.DataFrame:
        """The underlying triangle as a DataFrame (read-only copy)."""
        return self._data.copy()

    @property
    def shape(self) -> tuple[int, int]:
        """(n_origin_years, n_dev_lags)"""
        return self._data.shape

    @property
    def origin_years(self) -> pd.Index:
        """Accident years (row index)."""
        return self._data.index.copy()

    @property
    def dev_lags(self) -> pd.Index:
        """Development lags (column index)."""
        return self._data.columns.copy()

    @property
    def n_origins(self) -> int:
        """Number of accident years."""
        return self._data.shape[0]

    @property
    def n_devs(self) -> int:
        """Number of development lags."""
        return self._data.shape[1]

    @property
    def is_complete(self) -> bool:
        """True if the triangle has no missing (NaN) values."""
        return not self._data.isnull().any().any()

    @property
    def latest_diagonal(self) -> pd.Series:
        """
        The most recent known value for each accident year.

        For a standard upper-left triangle, this is the last non-NaN
        value in each row — the most recently observed development.
        """
        return self._data.apply(
            lambda row: row.dropna().iloc[-1] if not row.dropna().empty else np.nan,
            axis=1,
        )

    @property
    def latest_dev_lag(self) -> pd.Series:
        """
        The most recent development lag observed for each accident year.
        """
        return self._data.apply(
            lambda row: row.dropna().index[-1] if not row.dropna().empty else np.nan,
            axis=1,
        )

    # ------------------------------------------------------------------ #
    # Methods                                                             #
    # ------------------------------------------------------------------ #

    def to_incremental(self) -> "Triangle":
        """
        Convert cumulative triangle to incremental (period-over-period) losses.

        Returns
        -------
        Triangle
            A new Triangle with incremental values.
        """
        inc = self._data.diff(axis=1)
        inc.iloc[:, 0] = self._data.iloc[:, 0]
        return Triangle(inc)

    def link_ratios(self) -> pd.DataFrame:
        """
        Compute age-to-age link ratios for each cell.

        Returns a DataFrame of the same shape where each cell is
        value(lag+1) / value(lag). The last column is all NaN
        (no next period to develop to).

        Returns
        -------
        pd.DataFrame
        """
        ratios = self._data.copy()
        for i in range(len(self._data.columns) - 1):
            current_col = self._data.columns[i]
            next_col = self._data.columns[i + 1]
            ratios[current_col] = self._data[next_col] / self._data[current_col].replace(0, np.nan)
        ratios[self._data.columns[-1]] = np.nan
        return ratios

    def volume_weighted_factors(self) -> pd.Series:
        """
        Compute volume-weighted average development factors by lag.

        The volume-weighted factor at lag k is:
            sum(loss[lag k+1]) / sum(loss[lag k])
        summed over all accident years with data at both lags.

        Returns
        -------
        pd.Series indexed by development lag (excludes final lag).
        """
        factors = {}
        cols = list(self._data.columns)
        for i in range(len(cols) - 1):
            curr, nxt = cols[i], cols[i + 1]
            both = self._data[[curr, nxt]].dropna()
            if len(both) > 0 and both[curr].sum() > 0:
                factors[curr] = both[nxt].sum() / both[curr].sum()
            else:
                factors[curr] = np.nan
        return pd.Series(factors, name="vw_factor")

    def summary(self) -> pd.DataFrame:
        """
        Summary statistics for each development lag.

        Returns a DataFrame with columns: n_obs, mean, std, min, max
        for each development lag (non-NaN values only).

        Returns
        -------
        pd.DataFrame
        """
        return self._data.agg(["count", "mean", "std", "min", "max"]).T.rename(
            columns={"count": "n_obs"}
        )

    # ------------------------------------------------------------------ #
    # Dunder methods                                                      #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"Triangle(origins={self.n_origins}, "
            f"dev_lags={self.n_devs}, "
            f"complete={self.is_complete})"
        )

    def __str__(self) -> str:
        return self._data.to_string()

    def __getitem__(self, key):
        """Allow indexing directly into the underlying DataFrame."""
        return self._data[key]

    def __len__(self) -> int:
        return self.n_origins
