from abc import ABC, abstractmethod

from nafld.table.tables.static_table import StaticTable


class BaseTable(ABC):
    def __init__(self, table: StaticTable) -> None:
        self.table = table
        self.name = table.name
        self.path = table.path

    @abstractmethod
    def generate() -> None:
        raise NotImplementedError
