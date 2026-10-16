"""Dataset paths. Price data is NOT committed (repo is public):
  - GOOG / EURUSD / BTCUSD ship with the `backtesting` pip package and are loaded from there.
  - AAPL_daily.csv (Yahoo Finance) must be placed in exp-0/data/ yourself
    (columns: Date,Open,High,Low,Close,Volume). Scripts skip it if it is missing."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def sample(name):
    """name in {'GOOG', 'EURUSD', 'BTCUSD'} -> path inside the backtesting package."""
    import backtesting.test as t
    return Path(t.__file__).resolve().parent / f"{name}.csv"


AAPL = DATA / "AAPL_daily.csv"
