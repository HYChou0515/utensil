import os

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorClient

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
db_client: AsyncIOMotorClient = None


async def get_db_client() -> AsyncIOMotorClient:
    """Return database client instance."""
    return db_client


async def connect_db():
    """Create database connection."""
    global db_client
    db_client = AsyncIOMotorClient(MONGODB_URL)


async def close_db():
    """Close database connection."""
    global db_client
    if db_client is not None:
        db_client.close()


async def get_db():
    client = await get_db_client()
    db = client["loopflow-app"]
    return db


class DBSessionMixin:

    def __init__(self, db: get_db = Depends()):
        self.db = db
