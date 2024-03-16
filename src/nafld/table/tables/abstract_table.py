from abc import ABC, abstractmethod

from pandas import DataFrame


class AbstractTable(ABC):
    @abstractmethod
    def read(self) -> DataFrame:
        raise NotImplementedError

    @abstractmethod
    def write_parquet(self) -> DataFrame:
        raise NotImplementedError

    @abstractmethod
    def write_csv(self) -> DataFrame:
        raise NotImplementedError
