from enum import Enum


class RunListStatus(str, Enum):
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"

    def __str__(self) -> str:
        return str(self.value)
