from enum import Enum

INTERNAL_COLUMNS_PREFIX = "INTERNAL_"


class ColumnsEnum(str, Enum):
    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

    @classmethod
    def get_table_columns(cls) -> list[str]:
        return cls.get_static_columns() + cls.get_dynamic_columns()

    @classmethod
    def get_static_columns(cls) -> list[str]:
        return [column.value for column in cls if not column.name.startswith(INTERNAL_COLUMNS_PREFIX)]

    @classmethod
    def get_dynamic_columns(cls) -> list[str]:
        return []
