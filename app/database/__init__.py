import os

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorClient

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
db_client: AsyncIOMotorClient = None


async def get_db():
    global db_client
    if db_client is None:
        db_client = AsyncIOMotorClient(MONGODB_URL)
    db = db_client["loopflow-app"]
    return db


class DBSessionMixin:

    def __init__(self, db: AsyncIOMotorClient = Depends(get_db)):
        self.db = db
