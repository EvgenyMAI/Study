import os
import sys
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / "environment" / "psql_db.env"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def init_db():
    load_dotenv(dotenv_path=env_path)

    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        dbname=os.getenv("POSTGRES_DB"),
    )

    return conn