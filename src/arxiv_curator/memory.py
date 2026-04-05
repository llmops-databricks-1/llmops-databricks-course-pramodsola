"""Session memory management using Databricks Lakebase (PostgreSQL)."""

import json
from loguru import logger


class LakebaseMemory:
    """Manages conversation session memory using Lakebase (Databricks PostgreSQL).

    Stores and retrieves chat messages per session ID using a PostgreSQL table.
    Uses Databricks token authentication via psycopg.
    """

    def __init__(self, host: str, instance_name: str, port: int = 5432) -> None:
        """Initialize LakebaseMemory.

        Args:
            host: Lakebase read/write DNS hostname
            instance_name: Name of the Lakebase instance
            port: PostgreSQL port (default 5432)
        """
        self.host = host
        self.instance_name = instance_name
        self.port = port
        self._pool = None
        self._setup()

    def _get_token(self) -> str:
        """Get Databricks auth token from environment or SDK."""
        import os
        token = os.environ.get("DATABRICKS_TOKEN")
        if token:
            return token
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        # Try runtime token first
        try:
            from pyspark.dbutils import DBUtils
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            dbutils = DBUtils(spark)
            return dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        except Exception:
            pass
        return w.config.token or ""

    def _setup(self) -> None:
        """Create connection pool and ensure messages table exists."""
        try:
            import psycopg_pool
            import psycopg

            token = self._get_token()
            conninfo = (
                f"host={self.host} "
                f"port={self.port} "
                f"dbname=postgres "
                f"user=token "
                f"password={token} "
                f"sslmode=require"
            )
            self._pool = psycopg_pool.ConnectionPool(conninfo, min_size=1, max_size=5, open=True)
            self._create_table()
            logger.info(f"✓ LakebaseMemory connected to {self.host}")
        except Exception as e:
            logger.warning(f"⚠️ LakebaseMemory setup failed: {type(e).__name__}: {e}")
            self._pool = None

    def _create_table(self) -> None:
        """Create messages table if it doesn't exist."""
        if not self._pool:
            return
        with self._pool.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_messages (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_id ON session_messages(session_id)"
            )

    def save_messages(self, session_id: str, messages: list[dict]) -> None:
        """Save messages to a session.

        Args:
            session_id: Unique session identifier
            messages: List of message dicts with 'role' and 'content' keys
        """
        if not self._pool:
            logger.warning("LakebaseMemory not available — messages not persisted")
            return
        with self._pool.connection() as conn:
            for msg in messages:
                conn.execute(
                    "INSERT INTO session_messages (session_id, role, content) VALUES (%s, %s, %s)",
                    (session_id, msg["role"], msg["content"])
                )
        logger.debug(f"Saved {len(messages)} messages to session {session_id}")

    def load_messages(self, session_id: str) -> list[dict]:
        """Load all messages for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            List of message dicts ordered by creation time
        """
        if not self._pool:
            return []
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT role, content FROM session_messages WHERE session_id = %s ORDER BY id",
                (session_id,)
            ).fetchall()
        return [{"role": row[0], "content": row[1]} for row in rows]

    def delete_session(self, session_id: str) -> None:
        """Delete all messages for a session.

        Args:
            session_id: Unique session identifier
        """
        if not self._pool:
            return
        with self._pool.connection() as conn:
            conn.execute(
                "DELETE FROM session_messages WHERE session_id = %s", (session_id,)
            )
        logger.info(f"Deleted session: {session_id}")

    def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            self._pool.close()
