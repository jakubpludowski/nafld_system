from abc import ABC, abstractmethod

from nafld.table.tables.abstract_table import AbstractTable


class BaseTable(ABC):
    def __init__(self, table: AbstractTable) -> None:
        self.table = table

    @abstractmethod
    def generate() -> None:
        raise NotImplementedError
