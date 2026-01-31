"""Local filesystem storage implementation."""

from collections.abc import AsyncIterator
from pathlib import Path

import aiofiles
import aiofiles.os

from app.core.config import settings
from app.core.errors import StorageError
from app.core.logging import get_logger
from app.storage.storage_interface import StorageInterface

logger = get_logger(__name__)


class LocalStorage(StorageInterface):
    """Local filesystem storage backend."""

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize local storage.

        Args:
            base_dir: Base directory for storage. Defaults to settings.JOB_STORAGE_DIR.
        """
        self.base_dir = base_dir or settings.JOB_STORAGE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, path: str) -> Path:
        """Resolve a storage path to an absolute path.

        Args:
            path: The storage path.

        Returns:
            Absolute Path object.
        """
        resolved = self.base_dir / path
        # Security: ensure path stays within base directory
        try:
            resolved.resolve().relative_to(self.base_dir.resolve())
        except ValueError:
            raise StorageError(f"Invalid path: {path}")
        return resolved

    async def save(self, data: bytes, path: str) -> str:
        """Save data to local filesystem.

        Args:
            data: The data to save.
            path: The storage path relative to base_dir.

        Returns:
            The absolute path to the saved file.
        """
        file_path = self._resolve_path(path)

        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(file_path, "wb") as f:
                await f.write(data)

            logger.debug("File saved", path=str(file_path), size=len(data))
            return str(file_path)

        except Exception as e:
            logger.error("Failed to save file", path=str(file_path), error=str(e))
            raise StorageError(f"Failed to save file: {e}", path=path)

    async def save_stream(
        self, stream: AsyncIterator[bytes], path: str, chunk_size: int = 8192
    ) -> str:
        """Save a stream of data to local filesystem.

        Args:
            stream: Async iterator yielding bytes.
            path: The storage path relative to base_dir.
            chunk_size: Size of chunks to write.

        Returns:
            The absolute path to the saved file.
        """
        file_path = self._resolve_path(path)

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            total_size = 0

            async with aiofiles.open(file_path, "wb") as f:
                async for chunk in stream:
                    await f.write(chunk)
                    total_size += len(chunk)

            logger.debug("Stream saved", path=str(file_path), size=total_size)
            return str(file_path)

        except Exception as e:
            logger.error("Failed to save stream", path=str(file_path), error=str(e))
            raise StorageError(f"Failed to save stream: {e}", path=path)

    async def load(self, path: str) -> bytes:
        """Load data from local filesystem.

        Args:
            path: The storage path relative to base_dir.

        Returns:
            The loaded data.
        """
        file_path = self._resolve_path(path)

        if not file_path.exists():
            raise StorageError(f"File not found: {path}", path=path)

        try:
            async with aiofiles.open(file_path, "rb") as f:
                data = await f.read()

            logger.debug("File loaded", path=str(file_path), size=len(data))
            return data

        except Exception as e:
            logger.error("Failed to load file", path=str(file_path), error=str(e))
            raise StorageError(f"Failed to load file: {e}", path=path)

    async def delete(self, path: str) -> None:
        """Delete a file from local filesystem.

        Args:
            path: The storage path relative to base_dir.
        """
        file_path = self._resolve_path(path)

        if file_path.exists():
            try:
                await aiofiles.os.remove(file_path)
                logger.debug("File deleted", path=str(file_path))
            except Exception as e:
                logger.error("Failed to delete file", path=str(file_path), error=str(e))
                raise StorageError(f"Failed to delete file: {e}", path=path)

    async def delete_directory(self, path: str) -> None:
        """Delete a directory and its contents.

        Args:
            path: The directory path relative to base_dir.
        """
        dir_path = self._resolve_path(path)

        if dir_path.exists() and dir_path.is_dir():
            try:
                import shutil

                shutil.rmtree(dir_path)
                logger.debug("Directory deleted", path=str(dir_path))
            except Exception as e:
                logger.error("Failed to delete directory", path=str(dir_path), error=str(e))
                raise StorageError(f"Failed to delete directory: {e}", path=path)

    async def exists(self, path: str) -> bool:
        """Check if a path exists.

        Args:
            path: The storage path relative to base_dir.

        Returns:
            True if exists, False otherwise.
        """
        file_path = self._resolve_path(path)
        return file_path.exists()

    async def list_files(self, prefix: str = "") -> list[str]:
        """List files with a given prefix.

        Args:
            prefix: The path prefix to filter by.

        Returns:
            List of relative file paths.
        """
        search_path = self._resolve_path(prefix) if prefix else self.base_dir

        if not search_path.exists():
            return []

        files: list[str] = []
        if search_path.is_file():
            files.append(str(search_path.relative_to(self.base_dir)))
        else:
            for file_path in search_path.rglob("*"):
                if file_path.is_file():
                    files.append(str(file_path.relative_to(self.base_dir)))

        return sorted(files)

    async def get_size(self, path: str) -> int:
        """Get the size of a file.

        Args:
            path: The storage path relative to base_dir.

        Returns:
            File size in bytes.
        """
        file_path = self._resolve_path(path)

        if not file_path.exists():
            raise StorageError(f"File not found: {path}", path=path)

        return file_path.stat().st_size


def create_local_storage(base_dir: Path | None = None) -> LocalStorage:
    """Factory function to create LocalStorage.

    Args:
        base_dir: Base directory for storage.

    Returns:
        LocalStorage instance.
    """
    return LocalStorage(base_dir)
