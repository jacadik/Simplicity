#!/usr/bin/env python3
"""
Enhanced Database Diagnostic Tool for Paragraph Analyzer
This tool performs deep diagnosis and reports detailed information about the database.
"""

import os
import sqlite3
import sys
import time
import platform
import traceback
from datetime import datetime

# Default database path - change if your database is elsewhere
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paragraph_analyzer.db')

def get_sqlite_version():
    """Get SQLite version information"""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("SELECT sqlite_version()")
    version = cursor.fetchone()[0]
    conn.close()
    return version

def get_system_info():
    """Get system information"""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "sqlite_version": get_sqlite_version(),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return info

def get_table_schema(conn, table_name):
    """Get the actual schema of a table"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    
    cursor.execute(f"PRAGMA foreign_key_list({table_name})")
    foreign_keys = cursor.fetchall()
    
    cursor.execute(f"PRAGMA index_list({table_name})")
    indexes = cursor.fetchall()
    
    return {
        "columns": columns,
        "foreign_keys": foreign_keys,
        "indexes": indexes
    }

def execute_query_with_logging(conn, query, params=None):
    """Execute a SQL query and log details"""
    cursor = conn.cursor()
    start_time = time.time()
    result = None
    error = None
    
    try:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        if query.strip().upper().startswith("SELECT"):
            result = cursor.fetchall()
        else:
            result = {"rowcount": cursor.rowcount, "lastrowid": cursor.lastrowid}
    except Exception as e:
        error = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
    
    execution_time = time.time() - start_time
    
    log_entry = {
        "query": query,
        "params": params,
        "execution_time_ms": execution_time * 1000,
        "result": result if result is not None else None,
        "error": error
    }
    
    return log_entry

def test_delete_document(db_path, document_id):
    """Test deleting a document and log all steps"""
    print(f"\n===== TESTING DOCUMENT DELETION FOR ID {document_id} =====")
    logs = []
    
    try:
        # Connect with foreign keys enabled
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        # Enable foreign keys and get its status
        logs.append(execute_query_with_logging(conn, "PRAGMA foreign_keys = ON"))
        logs.append(execute_query_with_logging(conn, "PRAGMA foreign_keys"))
        
        # Check if document exists
        logs.append(execute_query_with_logging(
            conn, 
            "SELECT * FROM documents WHERE id = ?", 
            (document_id,)
        ))
        
        # Get associated paragraphs before deletion
        logs.append(execute_query_with_logging(
            conn, 
            "SELECT id FROM paragraphs WHERE document_id = ?", 
            (document_id,)
        ))
        
        # Begin transaction
        logs.append(execute_query_with_logging(conn, "BEGIN TRANSACTION"))
        
        # Try to delete document (should cascade)
        logs.append(execute_query_with_logging(
            conn, 
            "DELETE FROM documents WHERE id = ?", 
            (document_id,)
        ))
        
        # Check if paragraphs still exist
        logs.append(execute_query_with_logging(
            conn, 
            "SELECT id FROM paragraphs WHERE document_id = ?", 
            (document_id,)
        ))
        
        # Commit the transaction
        logs.append(execute_query_with_logging(conn, "COMMIT"))
        
        # Check again after commit
        logs.append(execute_query_with_logging(
            conn, 
            "SELECT id FROM paragraphs WHERE document_id = ?", 
            (document_id,)
        ))
        
        conn.close()
        
        print("\nDeletion Test Results:")
        
        # Print results in a readable format
        for i, log in enumerate(logs):
            print(f"\nStep {i+1}:")
            print(f"  Query: {log['query']}")
            if log['params']:
                print(f"  Params: {log['params']}")
            print(f"  Execution time: {log['execution_time_ms']:.2f} ms")
            
            if log['error']:
                print(f"  ERROR: {log['error']['error_type']}: {log['error']['error_message']}")
            elif log['result'] is not None:
                if isinstance(log['result'], list):
                    print(f"  Result: {len(log['result'])} rows")
                    if len(log['result']) > 0 and len(log['result']) < 10:
                        for row in log['result']:
                            print(f"    {row}")
                else:
                    print(f"  Result: {log['result']}")
        
        return logs
        
    except Exception as e:
        print(f"ERROR in test: {str(e)}")
        traceback.print_exc()
        if 'conn' in locals():
            conn.close()
        return logs

def test_manual_delete(db_path, document_id):
    """Test manually deleting a document and all associated data"""
    print(f"\n===== TESTING MANUAL CASCADE DELETION FOR ID {document_id} =====")
    logs = []
    
    try:
        # Connect with foreign keys disabled (to fully manual test)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        # Disable foreign keys for this test
        logs.append(execute_query_with_logging(conn, "PRAGMA foreign_keys = OFF"))
        
        # Begin transaction
        logs.append(execute_query_with_logging(conn, "BEGIN TRANSACTION"))
        
        # Get paragraph IDs
        paragraph_query = execute_query_with_logging(
            conn, 
            "SELECT id FROM paragraphs WHERE document_id = ?", 
            (document_id,)
        )
        logs.append(paragraph_query)
        
        if paragraph_query['result']:
            paragraph_ids = [row[0] for row in paragraph_query['result']]
            params_str = ','.join(['?' for _ in paragraph_ids])
            
            if paragraph_ids:
                # Delete similarity results
                logs.append(execute_query_with_logging(
                    conn,
                    f"DELETE FROM similarity_results WHERE paragraph1_id IN ({params_str}) OR paragraph2_id IN ({params_str})",
                    paragraph_ids + paragraph_ids
                ))
                
                # Delete paragraph tags
                logs.append(execute_query_with_logging(
                    conn,
                    f"DELETE FROM paragraph_tags WHERE paragraph_id IN ({params_str})",
                    paragraph_ids
                ))
                
                # Delete paragraphs
                logs.append(execute_query_with_logging(
                    conn,
                    f"DELETE FROM paragraphs WHERE id IN ({params_str})",
                    paragraph_ids
                ))
        
        # Delete document
        logs.append(execute_query_with_logging(
            conn,
            "DELETE FROM documents WHERE id = ?",
            (document_id,)
        ))
        
        # Check if paragraphs still exist
        logs.append(execute_query_with_logging(
            conn,
            "SELECT id FROM paragraphs WHERE document_id = ?",
            (document_id,)
        ))
        
        # Commit transaction
        logs.append(execute_query_with_logging(conn, "COMMIT"))
        
        conn.close()
        
        print("\nManual Deletion Test Results:")
        
        # Print results in a readable format
        for i, log in enumerate(logs):
            print(f"\nStep {i+1}:")
            print(f"  Query: {log['query']}")
            if log['params']:
                print(f"  Params: {log['params']}")
            print(f"  Execution time: {log['execution_time_ms']:.2f} ms")
            
            if log['error']:
                print(f"  ERROR: {log['error']['error_type']}: {log['error']['error_message']}")
            elif log['result'] is not None:
                if isinstance(log['result'], list):
                    print(f"  Result: {len(log['result'])} rows")
                    if len(log['result']) > 0 and len(log['result']) < 10:
                        for row in log['result']:
                            print(f"    {row}")
                else:
                    print(f"  Result: {log['result']}")
        
        return logs
        
    except Exception as e:
        print(f"ERROR in manual test: {str(e)}")
        traceback.print_exc()
        if 'conn' in locals():
            conn.close()
        return logs

def try_rebuild_schema(db_path):
    """Try to rebuild the schema by creating a new database"""
    print("\n===== ATTEMPTING DATABASE REBUILD =====")
    
    # Create a backup of the existing database
    backup_path = f"{db_path}.backup_{int(time.time())}"
    try:
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"Created backup at {backup_path}")
    except Exception as e:
        print(f"Warning: Failed to create backup: {str(e)}")
    
    try:
        # Connect to existing database to extract data
        old_conn = sqlite3.connect(db_path)
        old_conn.row_factory = sqlite3.Row
        
        # Export tags (to preserve them)
        old_cursor = old_conn.cursor()
        old_cursor.execute("SELECT * FROM tags")
        tags = [dict(row) for row in old_cursor.fetchall()]
        
        old_conn.close()
        
        # Rename or delete the original database
        temp_path = f"{db_path}.old"
        os.rename(db_path, temp_path)
        print(f"Renamed old database to {temp_path}")
        
        # Create a fresh database
        new_conn = sqlite3.connect(db_path)
        new_cursor = new_conn.cursor()
        
        # Enable foreign keys
        new_cursor.execute("PRAGMA foreign_keys = ON")
        
        # Create schema
        new_cursor.execute('''
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            upload_date TEXT NOT NULL
        )
        ''')
        
        new_cursor.execute('''
        CREATE TABLE paragraphs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            document_id INTEGER NOT NULL,
            paragraph_type TEXT NOT NULL,
            position INTEGER NOT NULL,
            header_content TEXT,
            FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
        )
        ''')
        
        new_cursor.execute('''
        CREATE TABLE tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            color TEXT NOT NULL
        )
        ''')
        
        new_cursor.execute('''
        CREATE TABLE paragraph_tags (
            paragraph_id INTEGER NOT NULL,
            tag_id INTEGER NOT NULL,
            PRIMARY KEY (paragraph_id, tag_id),
            FOREIGN KEY (paragraph_id) REFERENCES paragraphs (id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id) REFERENCES tags (id) ON DELETE CASCADE
        )
        ''')
        
        new_cursor.execute('''
        CREATE TABLE similarity_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paragraph1_id INTEGER NOT NULL,
            paragraph2_id INTEGER NOT NULL,
            similarity_score REAL NOT NULL,
            similarity_type TEXT NOT NULL,
            FOREIGN KEY (paragraph1_id) REFERENCES paragraphs (id) ON DELETE CASCADE,
            FOREIGN KEY (paragraph2_id) REFERENCES paragraphs (id) ON DELETE CASCADE
        )
        ''')
        
        # Create indexes
        new_cursor.execute('CREATE INDEX idx_paragraphs_document_id ON paragraphs (document_id)')
        new_cursor.execute('CREATE INDEX idx_similarity_paragraph1 ON similarity_results (paragraph1_id)')
        new_cursor.execute('CREATE INDEX idx_similarity_paragraph2 ON similarity_results (paragraph2_id)')
        
        # Import tags from old database
        for tag in tags:
            new_cursor.execute(
                'INSERT OR IGNORE INTO tags (id, name, color) VALUES (?, ?, ?)',
                (tag['id'], tag['name'], tag['color'])
            )
        
        new_conn.commit()
        new_conn.close()
        
        print("Successfully rebuilt database schema")
        print("Note: The new database contains only tag data. You will need to re-import your documents.")
        print(f"Your old data is preserved in {temp_path} if you need to recover it.")
        
        return True
    except Exception as e:
        print(f"ERROR rebuilding database: {str(e)}")
        traceback.print_exc()
        return False

def diagnose_database(db_path=DEFAULT_DB_PATH):
    """Run comprehensive diagnostics on the database"""
    print(f"\n===== COMPREHENSIVE DATABASE DIAGNOSIS: {db_path} =====")
    
    if not os.path.exists(db_path):
        print(f"ERROR: Database file does not exist at {db_path}")
        return
    
    # System information
    system_info = get_system_info()
    print("\nSystem Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # File information
    file_size = os.path.getsize(db_path)
    print(f"\nDatabase file size: {file_size/1024:.2f} KB")
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check various PRAGMA settings
        print("\nDatabase Settings:")
        for setting in ['foreign_keys', 'journal_mode', 'synchronous', 'secure_delete']:
            cursor.execute(f"PRAGMA {setting}")
            result = cursor.fetchone()[0]
            print(f"  {setting}: {result}")
        
        # Get table information
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"\nTables: {', '.join(tables)}")
        
        # Get row counts and detailed schema
        print("\nDetailed Table Information:")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            schema = get_table_schema(conn, table)
            
            print(f"\n  Table: {table} ({row_count} rows)")
            print("  Columns:")
            for col in schema['columns']:
                print(f"    - {col[1]} ({col[2]}){' PRIMARY KEY' if col[5] else ''}")
            
            if schema['foreign_keys']:
                print("  Foreign Keys:")
                for fk in schema['foreign_keys']:
                    print(f"    - {fk[3]} -> {fk[2]}.{fk[4]} (on delete: {fk[6] or 'NO ACTION'})")
            else:
                print("  Foreign Keys: None")
            
            if schema['indexes']:
                print("  Indexes:")
                for idx in schema['indexes']:
                    print(f"    - {idx[1]}")
        
        # Look for orphaned records
        print("\nChecking for orphaned records:")
        
        cursor.execute("""
            SELECT COUNT(*) FROM paragraphs p
            LEFT JOIN documents d ON p.document_id = d.id
            WHERE d.id IS NULL
        """)
        orphaned_paragraphs = cursor.fetchone()[0]
        print(f"  Orphaned paragraphs: {orphaned_paragraphs}")
        
        if orphaned_paragraphs > 0:
            cursor.execute("""
                SELECT p.id, p.document_id FROM paragraphs p
                LEFT JOIN documents d ON p.document_id = d.id
                WHERE d.id IS NULL
                LIMIT 10
            """)
            sample_orphans = cursor.fetchall()
            print("  Sample orphaned paragraphs:")
            for orphan in sample_orphans:
                print(f"    - Paragraph ID: {orphan[0]}, References Document ID: {orphan[1]}")
        
        # Check for integrity issues
        cursor.execute("PRAGMA integrity_check")
        integrity_result = cursor.fetchone()[0]
        print(f"\nIntegrity check: {integrity_result}")
        
        # Test a foreign key deletion if there's data
        cursor.execute("SELECT id FROM documents LIMIT 1")
        document = cursor.fetchone()
        if document:
            document_id = document[0]
            
            # Test cascade deletion
            test_delete_document(db_path, document_id)
            
            # If paragraphs still exist, try manual deletion
            cursor.execute("SELECT COUNT(*) FROM paragraphs WHERE document_id = ?", (document_id,))
            remaining_paragraphs = cursor.fetchone()[0]
            
            if remaining_paragraphs > 0:
                print(f"\nWARNING: {remaining_paragraphs} paragraphs remain after cascade deletion test")
                
                # Try manual deletion
                test_manual_delete(db_path, document_id)
        else:
            print("\nNo documents found to test deletion")
        
        conn.close()
        
        # Database rebuild option
        print("\nBased on the diagnosis, would you like to rebuild the database schema?")
        print("This will preserve your tags but you'll need to re-upload documents.")
        print("WARNING: This is a destructive operation. Your old database will be backed up.")
        choice = input("Rebuild database? (yes/no): ")
        if choice.lower().startswith('y'):
            try_rebuild_schema(db_path)
        
    except Exception as e:
        print(f"\nERROR during diagnosis: {str(e)}")
        traceback.print_exc()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    db_path = DEFAULT_DB_PATH
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    diagnose_database(db_path)
