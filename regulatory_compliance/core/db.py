import psycopg
from regulatory_compliance.core.config import settings


class Database:

    @staticmethod
    def get_connection():
        return psycopg.connect(settings.DATABASE_URL)
