"""MinIO (S3-compatible) storage implementation."""

import asyncio
import io
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from app.core.config import settings
from app.core.errors import StorageError
from app.core.logging import get_logger
from app.storage.storage_interface import StorageInterface

logger = get_logger(__name__)


class MinioStorage(StorageInterface):
    """MinIO storage backend."""

    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket: str | None = None,
        secure: bool | None = None,
    ) -> None:
        self.endpoint = endpoint or settings.MINIO_ENDPOINT
        self.access_key = access_key or settings.MINIO_ACCESS_KEY
        self.secret_key = secret_key or settings.MINIO_SECRET_KEY
        self.bucket = bucket or settings.MINIO_BUCKET
        self.secure = settings.MINIO_SECURE if secure is None else secure

        if not self.access_key or not self.secret_key:
            raise StorageError("MinIO credentials are required")

        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
        )

    async def ensure_bucket(self) -> None:
        """Ensure bucket exists."""
        try:
            exists = await asyncio.to_thread(self.client.bucket_exists, self.bucket)
            if not exists:
                await asyncio.to_thread(self.client.make_bucket, self.bucket)
                logger.info("MinIO bucket created", bucket=self.bucket)
        except S3Error as e:
            logger.error("Failed to ensure MinIO bucket", error=str(e))
            raise StorageError(f"Failed to ensure MinIO bucket: {e}")

    def _build_url(self, object_name: str) -> str:
        scheme = "https" if self.secure else "http"
        return f"{scheme}://{self.endpoint}/{self.bucket}/{object_name}"

    async def save(self, data: bytes, path: str) -> str:
        try:
            data_stream = io.BytesIO(data)
            await asyncio.to_thread(
                self.client.put_object,
                self.bucket,
                path,
                data_stream,
                length=len(data),
                content_type="application/octet-stream",
            )
            return self._build_url(path)
        except S3Error as e:
            logger.error("Failed to save to MinIO", path=path, error=str(e))
            raise StorageError(f"Failed to save to MinIO: {e}", path=path)

    async def save_file(self, file_path: Path, object_name: str) -> str:
        try:
            await asyncio.to_thread(
                self.client.fput_object,
                self.bucket,
                object_name,
                str(file_path),
            )
            return self._build_url(object_name)
        except S3Error as e:
            logger.error("Failed to upload file to MinIO", path=object_name, error=str(e))
            raise StorageError(f"Failed to upload file to MinIO: {e}", path=object_name)

    async def load(self, path: str) -> bytes:
        try:
            response = await asyncio.to_thread(self.client.get_object, self.bucket, path)
            try:
                data = await asyncio.to_thread(response.read)
            finally:
                response.close()
                response.release_conn()
            return data
        except S3Error as e:
            logger.error("Failed to load from MinIO", path=path, error=str(e))
            raise StorageError(f"Failed to load from MinIO: {e}", path=path)

    async def delete(self, path: str) -> None:
        try:
            await asyncio.to_thread(self.client.remove_object, self.bucket, path)
        except S3Error as e:
            logger.error("Failed to delete from MinIO", path=path, error=str(e))
            raise StorageError(f"Failed to delete from MinIO: {e}", path=path)

    async def exists(self, path: str) -> bool:
        try:
            await asyncio.to_thread(self.client.stat_object, self.bucket, path)
            return True
        except S3Error:
            return False

    async def list_files(self, prefix: str = "") -> list[str]:
        try:
            objects = await asyncio.to_thread(
                lambda: list(self.client.list_objects(self.bucket, prefix=prefix))
            )
            return [obj.object_name for obj in objects]
        except S3Error as e:
            logger.error("Failed to list MinIO objects", error=str(e))
            raise StorageError(f"Failed to list MinIO objects: {e}")


def create_minio_storage() -> MinioStorage:
    """Factory function to create MinioStorage."""
    return MinioStorage()
