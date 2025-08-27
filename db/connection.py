import psycopg2


def connect_db():
    return psycopg2.connect(
        dbname="visaocomputacional",
        user="compvis",
        password="compvis",
        host="localhost",
        port=5432,
    )
