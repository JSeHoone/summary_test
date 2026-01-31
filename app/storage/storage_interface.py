"""Abstract interface for storage backends."""

from abc import ABC, abstractmethod


class StorageInterface(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    async def save(self, data: bytes, path: str) -> str:
        """Save data to storage.

        Args:
            data: The data to save.
            path: The storage path/key.

        Returns:
            The full path/URL to the saved file.
        """
        pass

    @abstractmethod
    async def load(self, path: str) -> bytes:
        """Load data from storage.

        Args:
            path: The storage path/key.

        Returns:
            The loaded data.
        """
        pass

    @abstractmethod
    async def delete(self, path: str) -> None:
        """Delete data from storage.

        Args:
            path: The storage path/key.
        """
        pass

    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Check if a path exists in storage.

        Args:
            path: The storage path/key.

        Returns:
            True if exists, False otherwise.
        """
        pass

    @abstractmethod
    async def list_files(self, prefix: str = "") -> list[str]:
        """List files with a given prefix.

        Args:
            prefix: The path prefix to filter by.

        Returns:
            List of file paths/keys.
        """
        pass
