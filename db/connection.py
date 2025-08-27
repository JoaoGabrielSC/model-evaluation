import psycopg2


def connect_db():
    return psycopg2.connect(
        dbname="visaocomputacional",
        user="const",
        password="const",
        host="localhost",
        port=5432,
    )
