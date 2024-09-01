"""
Implements data loading environment for following data types:
    - Brownian motion
    - AR(1) process
    - S&P500 log-returns
    - FOREX EUR/USD log-returns
"""

import torch
import math
import numpy as np
import pandas as pd
import yfinance as yf
from utils import *


class Data:
    """
    The parent class for data classes
    """

    def __init__(self, n_lags: int):
        self.n_lags = n_lags

    def generate(self, *kwargs):
        pass


class BM(Data):
    """
    Class implementing generation of Brownian motion paths
    """

    def __init__(self, n_lags: int, drift: float = 0.0, std: float = 1.0, dim: int = 1, T: float = 1.0):
        super().__init__(n_lags)
        self.drift = drift
        self.std = std
        self.dim = dim
        self.h = T / n_lags
        self.scaler = IDScaler()

    def generate(self, samples: int) -> torch.tensor:
        path = torch.zeros([samples, self.n_lags, self.dim])
        path[:, 1:, :] = self.drift * self.h + math.sqrt(self.h) * self.std * torch.randn(samples, self.n_lags - 1,
                                                                                          self.dim)
        return torch.cumsum(path, 1)
    

def simulate_BM(n_sample, dt, n_timestep):
    noise = torch.randn(size=(n_sample, n_timestep))
    paths_incr = noise * torch.sqrt(torch.tensor(dt))
    paths = torch.cumsum(paths_incr, axis=1)
    BM_paths = torch.cat([torch.zeros((n_sample, 1)), paths], axis=1)
    BM_paths = BM_paths[..., None]
    return BM_paths


def simulate_BS(n_sample, dt, n_timestep, mu, sigma):
    time_grid = torch.linspace(0, dt * n_timestep, n_timestep + 1)
    time_paths = time_grid.expand([n_sample, n_timestep + 1])[..., None]
    BM_paths = simulate_BM(n_sample, dt, n_timestep)
    BS_paths = torch.exp(sigma * BM_paths + (mu - 0.5 * sigma**2) * time_paths)
    return BS_paths

class BS(Data):
    """
    Class implementing generation of Brownian motion paths
    """

    def __init__(self, n_lags: int, drift: float = 0.0, std: float = 1.0, dim: int = 1, T: float = 1.0):
        super().__init__(n_lags)
        self.drift = drift
        self.std = std
        self.dim = dim
        self.n_lags = n_lags
        self.h = T / n_lags
        self.scaler = IDScaler()

    def generate(self, samples: int) -> torch.tensor:
        n_sample = samples
        dt = self.h
        n_timestep = self.n_lags-1
        path = simulate_BS(n_sample, dt, n_timestep, self.drift, self.std)
        return path


class AR(Data):
    """
    Class implementing generation of paths of AR(1) process
    """

    def __init__(self, n_lags: int, phi: float, std: float = 1.0, dim: int = 1):
        super().__init__(n_lags)
        self.phi = phi
        self.std = std
        self.dim = dim
        self.scaler = Scaler()

    def generate(self, samples: int) -> torch.tensor:
        paths = torch.zeros([samples, self.n_lags, self.dim])
        for i in range(1, self.n_lags):
            paths[:, i, :] = self.phi * paths[:, i - 1, :] + self.std * torch.randn(samples, self.dim)
        paths = self.scaler.transform(paths)
        return paths


class SP500(Data):
    """
    Class loading S&P 500 log-returns and applying a rolling window
    """

    def __init__(self, n_lags: int, start: str = "2005-01-01", end: str = "2023-10-31"):
        super().__init__(n_lags)
        self.start = start
        self.end = end
        self.scaler = Scaler()

    def generate(self) -> torch.tensor:
        data = yf.download("SPY", start=self.start, end=self.end)
        log_returns = (np.log(data["Close"]) - np.log(data["Close"].shift(1)))[1:].to_numpy().reshape(-1, 1)
        log_returns = torch.from_numpy(log_returns).float().unsqueeze(0)
        log_returns = self.scaler.transform(log_returns)
        paths = rolling_window(log_returns, self.n_lags)
        return paths


class FOREX(Data):
    """
    Class loading FOREX EUR/USD log-returns and applying a rolling window
    """

    def __init__(self, n_lags: int):
        super().__init__(n_lags)
        self.scaler = Scaler()

    def generate(self) -> torch.tensor:
        data = pd.read_csv("/content/drive/MyDrive/ColabNotebooks/Code_RSig_Gen/Unconditional_Generator/EURUSD1.csv",
                           sep='\t')
        data.columns = ["Date", "Open", "High", "Low", "Close", "Vol"]
        log_returns = (np.log(data.Close) - np.log(data.Close).shift(1))[1:].to_numpy().reshape(-1, 1)
        log_returns = torch.from_numpy(log_returns).float().unsqueeze(0)
        log_returns = self.scaler.transform(log_returns)
        paths = rolling_window(log_returns, self.n_lags)
        return paths


class Scaler:
    """
    Class for the standardisation of data
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.shift_by = None

    def transform(self, x: torch.tensor) -> torch.tensor:
        self.mean = x.mean()
        self.std = x.std()
        return (x - self.mean) / self.std

    def inverse(self, x: torch.tensor) -> torch.tensor:
        return x * self.std + self.mean


class IDScaler:
    """
    Class for scaler applying the identity function
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.shift_by = None

    def transform(self, x: torch.tensor) -> torch.tensor:
        return x

    def inverse(self, x: torch.tensor) -> torch.tensor:
        return x


"""
Function returning the sample paths for given data ID
"""

def get_data(id: str) -> torch.tensor:
    data, paths = None, None
    if id == "BM":
        data = BM(N_LAGS, DRIFT_BM, STD_BM, DATA_DIM, DATA_T)
        paths = data.generate(SAMPLES_BM)
    elif id == "GBM":
        data = GBM(N_LAGS, DRIFT_GBM, STD_GBM, INIT_GBM, DATA_DIM)
        paths = data.generate(SAMPLES_BM)
    elif id == "SP500":
        data = SP500(N_LAGS)
        paths = data.generate()
    elif id == "AR":
        data = AR(N_LAGS, PHI)
        paths = data.generate(SAMPLES_AR)
    elif id == "FOREX":
        data = FOREX(N_LAGS)
        paths = data.generate()
    elif id == "BS":
        data = BS(N_LAGS, DRIFT_BS, STD_BS, DATA_DIM, DATA_T)
        paths = data.generate(SAMPLES_BS)
    return [data, train_test_split(paths)]
