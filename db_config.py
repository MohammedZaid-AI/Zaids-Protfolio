"""
Database configuration abstraction for SQLite ↔ PostgreSQL

This module provides a unified interface for database operations that works
with both SQLite (for local development) and PostgreSQL (for production).
"""

import os
import sqlite3

# Check if we're using PostgreSQL (production) or SQLite (development)
POSTGRES_URL = os.environ.get("POSTGRES_URL")
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "data.db")

class PostgresCursorWrapper:
    """Cursor wrapper to translate SQLite syntax/placeholders to PostgreSQL equivalents"""
    def __init__(self, cursor):
        self._cursor = cursor
        
    def execute(self, query, params=None):
        # Translate SQLite-style placeholder '?' to PostgreSQL '%s'
        query = query.replace('?', '%s')
        # Translate SQLite-specific datetime() function calls to standard column sorting
        query = query.replace('datetime(created_at)', 'created_at')
        query = query.replace('datetime(timestamp)', 'timestamp')
        if params is not None:
            return self._cursor.execute(query, params)
        else:
            return self._cursor.execute(query)
            
    def executemany(self, query, params_list):
        query = query.replace('?', '%s')
        query = query.replace('datetime(created_at)', 'created_at')
        query = query.replace('datetime(timestamp)', 'timestamp')
        return self._cursor.executemany(query, params_list)
        
    def __getattr__(self, name):
        return getattr(self._cursor, name)

class PostgresConnectionWrapper:
    """Connection wrapper to make psycopg2 connections behave like sqlite3 connections"""
    def __init__(self, conn):
        self._conn = conn
        
    def cursor(self, *args, **kwargs):
        from psycopg2.extras import RealDictCursor
        kwargs.setdefault('cursor_factory', RealDictCursor)
        raw_cursor = self._conn.cursor(*args, **kwargs)
        return PostgresCursorWrapper(raw_cursor)
        
    def commit(self):
        return self._conn.commit()
        
    def rollback(self):
        return self._conn.rollback()
        
    def close(self):
        return self._conn.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback()
        else:
            self.commit()
        self.close()
        
    def __getattr__(self, name):
        return getattr(self._conn, name)

def get_db():
    """Retrieve database connection compatible with both SQLite and PostgreSQL"""
    if POSTGRES_URL:
        import psycopg2
        conn = psycopg2.connect(POSTGRES_URL)
        return PostgresConnectionWrapper(conn)
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

def init_db():
    """Initialize database tables (PostgreSQL serial keys or SQLite autoincrement keys)"""
    if POSTGRES_URL:
        # Production: Use PostgreSQL
        import psycopg2
        conn = psycopg2.connect(POSTGRES_URL)
        try:
            cur = conn.cursor()
            
            # Create tables (with PostgreSQL serial primary keys)
            tables = [
                """
                CREATE TABLE IF NOT EXISTS profile (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    tagline TEXT,
                    about TEXT,
                    github TEXT,
                    twitter TEXT,
                    linkedin TEXT,
                    website TEXT,
                    avatar_path TEXT,
                    avatar_back_path TEXT
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    image_path TEXT,
                    link TEXT,
                    github_link TEXT,
                    created_at TEXT
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS experience (
                    id SERIAL PRIMARY KEY,
                    role TEXT,
                    company TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    description TEXT,
                    logo_path TEXT
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS skills (
                    id SERIAL PRIMARY KEY,
                    name TEXT,
                    exploring INTEGER DEFAULT 0
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY,
                    knowledge_pdf_path TEXT
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    message TEXT,
                    timestamp TEXT
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS research_papers (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    publication TEXT,
                    date TEXT,
                    link TEXT,
                    description TEXT
                )
                """
            ]
            
            for table_sql in tables:
                cur.execute(table_sql)
            conn.commit()
                
            # Insert initial data if tables are empty
            from psycopg2.extras import RealDictCursor
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("SELECT COUNT(*) AS c FROM profile")
            if cur.fetchone()["c"] == 0:
                cur.execute(
                    "INSERT INTO profile (id, name, tagline, about, github, twitter, linkedin, website, avatar_path, avatar_back_path) VALUES (1, '', '', '', '', '', '', '', '', '')"
                )
                
            cur.execute("SELECT COUNT(*) AS c FROM settings")
            if cur.fetchone()["c"] == 0:
                cur.execute("INSERT INTO settings (id, knowledge_pdf_path) VALUES (1, '')")
            
            conn.commit()
        except Exception as e:
            print(f"PostgreSQL Database init skipped or failed: {e}")
            conn.rollback()
        finally:
            conn.close()
            
    else:
        # Development: Use SQLite
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            
            # Create tables
            tables = [
                """
                CREATE TABLE IF NOT EXISTS profile (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    tagline TEXT,
                    about TEXT,
                    github TEXT,
                    twitter TEXT,
                    linkedin TEXT,
                    website TEXT,
                    avatar_path TEXT,
                    avatar_back_path TEXT
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    description TEXT,
                    image_path TEXT,
                    link TEXT,
                    github_link TEXT,
                    created_at TEXT
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS experience (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT,
                    company TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    description TEXT,
                    logo_path TEXT
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    exploring INTEGER DEFAULT 0
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY,
                    knowledge_pdf_path TEXT
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    email TEXT,
                    message TEXT,
                    timestamp TEXT
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS research_papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    publication TEXT,
                    date TEXT,
                    link TEXT,
                    description TEXT
                )
                """
            ]
            
            for table_sql in tables:
                cur.execute(table_sql)
            conn.commit()
                
            cur.execute("SELECT COUNT(*) AS c FROM profile")
            if cur.fetchone()["c"] == 0:
                cur.execute(
                    "INSERT INTO profile (id, name, tagline, about, github, twitter, linkedin, website, avatar_path, avatar_back_path) VALUES (1, '', '', '', '', '', '', '', '', '')"
                )
                
            cur.execute("SELECT COUNT(*) AS c FROM settings")
            if cur.fetchone()["c"] == 0:
                cur.execute("INSERT INTO settings (id, knowledge_pdf_path) VALUES (1, '')")
            
            conn.commit()
        except Exception as e:
            print(f"SQLite Database init skipped or failed: {e}")
            conn.rollback()
        finally:
            conn.close()

def execute_query(query, params=()):
    """Execute a query and return results"""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        if query.strip().upper().startswith("SELECT"):
            return cur.fetchall()
        return None
    finally:
        conn.close()

def execute_update(query, params=()):
    """Execute an update/insert/delete query and return last inserted ID"""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
        if POSTGRES_URL:
            return getattr(cur, 'lastrowid', None)
        else:
            return cur.lastrowid
    finally:
        conn.close()

# Export the main functions
__all__ = ['get_db', 'init_db', 'execute_query', 'execute_update']