import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

from motor_insurance_risk.config import DB_CONNECTION_STRING, MOTOR_EMBEDDINGS_TABLE
from motor_insurance_risk.utils.logger import logger

class DatabaseConnection:
    def __init__(self):
        self.conn = None
        self.cur = None

    def connect(self):
        """Establish database connection"""
        logger.info("Opening DB connection")
        self.conn = psycopg2.connect(DB_CONNECTION_STRING)
        self.cur = self.conn.cursor()
        register_vector(self.conn)
        return self.conn, self.cur

    def close(self):
        """Close database connection"""
        logger.info("DB connection closed")
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

    def create_tables(self):
        """Create necessary tables if they don't exist"""
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {MOTOR_EMBEDDINGS_TABLE} (
            id bigserial primary key, 
            document text,
            driver_id integer,
            vehicle_id integer,
            policy_id integer,
            underwriting_decision text,
            risk_class text,
            reason_for_decline text,
            content text,
            tokens integer,
            embedding vector(1536)
        );
        """
        self.cur.execute(create_table_query)
        self.conn.commit()

    def batch_insert_embeddings(self, data_list):
        """Insert batch of embeddings data"""
        query = f"""
            INSERT INTO {MOTOR_EMBEDDINGS_TABLE}
            (document, driver_id, vehicle_id, policy_id, underwriting_decision, 
             risk_class, reason_for_decline, content, tokens, embedding)
            VALUES %s
        """
        execute_values(self.cur, query, data_list)
        self.conn.commit()

    def get_similar_documents(self, embedding_array, limit=5):
        """Retrieve similar documents using vector similarity search"""
        query = f"""
            SELECT content 
            FROM {MOTOR_EMBEDDINGS_TABLE} 
            ORDER BY embedding <=> %s::vector 
            LIMIT %s
        """
        self.cur.execute(query, (embedding_array, limit))
        return self.cur.fetchall()