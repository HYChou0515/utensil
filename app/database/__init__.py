import os

import pymongo
from fastapi import Depends

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")


def get_db():
    client = pymongo.MongoClient(MONGODB_URL)
    db = client["loopflow-app"]
    return db


class DBSessionMixin:

    def __init__(self, db: get_db = Depends()):
        self.db = db
