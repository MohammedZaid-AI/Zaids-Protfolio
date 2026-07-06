"""
Migration script to move data from SQLite to PostgreSQL

This script migrates all data from the local SQLite database to PostgreSQL
for production deployment on Vercel.
"""

import os
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor

# Get PostgreSQL connection string from environment
POSTGRES_URL = os.environ.get("POSTGRES_URL")

if not POSTGRES_URL:
    print("Error: POSTGRES_URL environment variable not set")
    print("Please set POSTGRES_URL from your Vercel Postgres database settings")
    exit(1)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "data.db")

def migrate_data():
    """Migrate all data from SQLite to PostgreSQL"""
    print("Starting migration from SQLite to PostgreSQL...")
    print(f"SQLite database: {DB_PATH}")
    print(f"PostgreSQL URL: {POSTGRES_URL[:50]}...")
    
    # Connect to SQLite
    sqlite_conn = sqlite3.connect(DB_PATH)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cur = sqlite_conn.cursor()
    
    # Connect to PostgreSQL
    pg_conn = psycopg2.connect(POSTGRES_URL, cursor_factory=RealDictCursor)
    pg_cur = pg_conn.cursor()
    
    try:
        # Disable foreign key checks in PostgreSQL
        pg_cur.execute("SET CONSTRAINTS ALL DEFERRED")
        
        # Get all data from SQLite tables
        tables = [
            "profile", "projects", "experience", "skills", 
            "settings", "messages", "research_papers"
        ]
        
        for table in tables:
            print(f"\nMigrating table: {table}")
            
            # Get all rows from SQLite
            sqlite_cur.execute(f"SELECT * FROM {table}")
            rows = sqlite_cur.fetchall()
            
            if not rows:
                print(f"  No data found in {table}")
                continue
                
            print(f"  Found {len(rows)} rows")
            
            # Get column names
            column_names = [description[0] for description in sqlite_cur.description]
            
            # Prepare insert query for PostgreSQL
            placeholders = ", ".join(["%s"] * len(column_names))
            insert_query = f"INSERT INTO {table} ({', '.join(column_names)}) VALUES ({placeholders})"
            
            # Convert SQLite rows to PostgreSQL format
            data_to_insert = []
            for row in rows:
                # Convert sqlite3.Row to dict
                row_dict = dict(row)
                
                # Convert any SQLite-specific types to PostgreSQL-compatible types
                for key, value in row_dict.items():
                    if value is None:
                        row_dict[key] = None
                    elif isinstance(value, str):
                        # Keep as is
                        pass
                    elif isinstance(value, (int, float)):
                        # Keep as is
                        pass
                    elif isinstance(value, bytes):
                        # Convert bytes to string
                        row_dict[key] = value.decode('utf-8', errors='ignore')
                
                data_to_insert.append(tuple(row_dict[col] for col in column_names))
            
            # Insert data into PostgreSQL in batches
            batch_size = 100
            for i in range(0, len(data_to_insert), batch_size):
                batch = data_to_insert[i:i + batch_size]
                pg_cur.executemany(insert_query, batch)
                print(f"  Inserted batch of {len(batch)} rows")
        
        # Re-enable foreign key checks
        pg_cur.execute("SET CONSTRAINTS ALL IMMEDIATE")
        
        print("\nMigration completed successfully!")
        
    except Exception as e:
        print(f"\nError during migration: {e}")
        pg_conn.rollback()
        raise
        
    finally:
        # Clean up connections
        sqlite_conn.close()
        pg_conn.close()

if __name__ == "__main__":
    migrate_data()