import os
import redis
from dotenv import load_dotenv

load_dotenv("../environment/env.env")

class RedisClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisClient, cls).__new__(cls)
            cls._instance.init_connection()
        return cls._instance

    def init_connection(self):
        self.connection = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=os.getenv("REDIS_PORT"),
            db=os.getenv("REDIS_DB"),
            password=os.getenv("REDIS_PASSWORD") or None,
            decode_responses=True
        )

    def get_connection(self):
        return self.connection

redis_client = RedisClient()