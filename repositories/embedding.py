from typing import List, Union
from db.connection import connect_db


class FaceEmbeddingRepository:
    def __init__(self):
        self.conn = connect_db()
        self.cursor = self.conn.cursor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cursor.close()
        self.conn.commit()
        self.conn.close()

    def create_table(self, table_name: str, vector_dimension: int):
        self.cursor.execute(
            f"""
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                embedding VECTOR({vector_dimension}),
                model_type TEXT
            );
            """
        )

    def search_similar(
        self,
        table_name: str,
        embedding: List[float],
        operator: str,
        limit: int = 1,
        threshold: Union[float, None] = None,
    ):
        vector_dim = len(embedding)
        query = f"""
        WITH input_vector AS (
            SELECT %s::vector({vector_dim}) AS v
        )
        SELECT name, 1 - embedding {operator} v AS distance
        FROM {table_name}, input_vector
        ORDER BY embedding {operator} v
        LIMIT %s;
        """
        embedding_str = f"[{', '.join(map(str, embedding))}]"
        self.cursor.execute(query, (embedding_str, limit))
        results = self.cursor.fetchall()

        if threshold is not None:
            results = [(name, dist) for name, dist in results if dist < threshold]
        return results

    def insert_embedding(
        self, table_name: str, person: str, embedding: List[float], model_name: str
    ):
        self.cursor.execute(
            f"""
            INSERT INTO {table_name} (name, embedding, model_type)
            VALUES (%s, %s, %s)
            """,
            (person, embedding, model_name),
        )

    def delete_table(self, table_name: str):
        table = f"face_embeddings_{table_name}"
        print(f"Deleting table: {table}")
        self.cursor.execute(f"DROP TABLE IF EXISTS {table};")
