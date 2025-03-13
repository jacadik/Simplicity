#!/usr/bin/env python3
"""
Database Fix Script for Paragraph Analyzer
Run this script to enable foreign keys on the database.
"""

import os
import sqlite3

# Default database path - change if your database is elsewhere
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paragraph_analyzer.db')

def fix_database(db_path=DEFAULT_DB_PATH):
    """Fix the database by enabling foreign keys and verifying structure"""
    print(f"\n===== FIXING DATABASE: {db_path} =====")
    
    if not os.path.exists(db_path):
        print(f"ERROR: Database file does not exist at {db_path}")
        return
    
    print(f"Database file exists ({os.path.getsize(db_path)/1024:.2f} KB)")
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check current foreign key status
        cursor.execute("PRAGMA foreign_keys")
        fk_status = cursor.fetchone()[0]
        print(f"Current foreign keys status: {'ENABLED' if fk_status == 1 else 'DISABLED'}")
        
        # Try to enable foreign keys permanently
        print("Enabling foreign keys...")
        
        # 1. Create a new connection with foreign keys enabled
        conn.close()
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        
        # 2. Check if the foreign keys were enabled
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys")
        fk_status = cursor.fetchone()[0]
        print(f"New foreign keys status: {'ENABLED' if fk_status == 1 else 'DISABLED'}")
        
        # Check journal mode - WAL mode is recommended for better performance
        cursor.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()[0]
        print(f"Current journal mode: {journal_mode}")
        
        # Set to WAL mode for better performance
        if journal_mode.upper() != 'WAL':
            print("Setting journal mode to WAL for better performance...")
            cursor.execute("PRAGMA journal_mode = WAL")
            journal_mode = cursor.fetchone()[0]
            print(f"New journal mode: {journal_mode}")
        
        # Run ANALYZE to optimize database performance
        print("Running ANALYZE to optimize database...")
        cursor.execute("ANALYZE")
        
        # Perform integrity check
        print("Performing integrity check...")
        cursor.execute("PRAGMA integrity_check")
        integrity = cursor.fetchone()[0]
        print(f"Integrity check result: {integrity}")
        
        # Get table row counts
        print("\nTable row counts:")
        for table in ['documents', 'paragraphs', 'tags', 'paragraph_tags', 'similarity_results']:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  - {table}: {count} rows")
        
        # Check for orphaned records
        cursor.execute("""
            SELECT COUNT(*) FROM paragraphs p
            LEFT JOIN documents d ON p.document_id = d.id
            WHERE d.id IS NULL
        """)
        orphaned_paragraphs = cursor.fetchone()[0]
        print(f"\nOrphaned paragraphs (not linked to valid document): {orphaned_paragraphs}")
        
        if orphaned_paragraphs > 0:
            print("Would you like to delete orphaned paragraphs? (yes/no)")
            choice = input().lower()
            if choice.startswith('y'):
                cursor.execute("""
                    DELETE FROM paragraphs 
                    WHERE document_id NOT IN (SELECT id FROM documents)
                """)
                conn.commit()
                print(f"Deleted {orphaned_paragraphs} orphaned paragraphs")
        
        conn.close()
        
        print("\nDatabase fix complete. You'll need to manually enable foreign keys for each connection.")
        print("To fix your app permanently, you need to modify the database_manager.py file to enable foreign keys.")
        print("See the instructions below.\n")
        
        print("IMPORTANT: Future Steps")
        print("1. Add '_get_connection()' to database_manager.py with foreign keys enabled")
        print("2. Replace all 'sqlite3.connect()' calls with '_get_connection()'")
        print("3. Use explicit deletion logic for 'delete_document' and 'clear_database' methods")
        
    except Exception as e:
        print(f"ERROR during fix: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    fix_database()
