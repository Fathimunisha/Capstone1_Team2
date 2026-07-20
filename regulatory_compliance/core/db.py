import psycopg
from regulatory_compliance.core.config import settings


class Database:
    """
    PostgreSQL database connection manager.
    """

    @staticmethod
    def get_connection():

        try:

            conn = psycopg.connect(settings.DATABASE_URL)

            return conn

        except Exception as e:

            print("Database connection failed:", e)

            raise
