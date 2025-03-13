#!/usr/bin/env python3
"""
Database Diagnostic Script for Paragraph Analyzer
Run this script to check the structure of your database and diagnose issues.
"""

import os
import sqlite3
import sys

# Default database path - change if your database is elsewhere
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paragraph_analyzer.db')

def check_database(db_path=DEFAULT_DB_PATH):
    """Check the database structure and diagnose issues"""
    print(f"\n===== CHECKING DATABASE: {db_path} =====")
    
    if not os.path.exists(db_path):
        print(f"ERROR: Database file does not exist at {db_path}")
        return
    
    print(f"Database file exists ({os.path.getsize(db_path)/1024:.2f} KB)")
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check foreign key status
        cursor.execute("PRAGMA foreign_keys")
        fk_status = cursor.fetchone()[0]
        print(f"Foreign keys status: {'ENABLED' if fk_status == 1 else 'DISABLED'}")
        
        # Get table list
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Tables found: {', '.join(tables)}")
        
        # Check each table's structure and data
        for table in tables:
            # Get table structure
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            print(f"\nTable: {table} ({row_count} rows)")
            print("Columns:")
            for col in columns:
                print(f"  - {col[1]} ({col[2]}){' PRIMARY KEY' if col[5] else ''}")
            
            # Check foreign keys for this table
            cursor.execute(f"PRAGMA foreign_key_list({table})")
            fks = cursor.fetchall()
            
            if fks:
                print("Foreign Keys:")
                for fk in fks:
                    print(f"  - {fk[3]} -> {fk[2]}.{fk[4]} (on delete: {fk[6] or 'NO ACTION'})")
            else:
                print("Foreign Keys: None")
        
        # If documents exist, check for orphaned data
        if "documents" in tables and "paragraphs" in tables:
            cursor.execute("SELECT id FROM documents")
            doc_ids = [row[0] for row in cursor.fetchall()]
            
            if doc_ids:
                placeholders = ','.join(['?' for _ in doc_ids])
                cursor.execute(f"SELECT COUNT(*) FROM paragraphs WHERE document_id NOT IN ({placeholders})", doc_ids)
                orphaned_paragraphs = cursor.fetchone()[0]
                print(f"\nOrphaned paragraphs (not linked to valid document): {orphaned_paragraphs}")
            
            if "similarity_results" in tables and "paragraphs" in tables:
                cursor.execute("SELECT id FROM paragraphs")
                para_ids = [row[0] for row in cursor.fetchall()]
                
                if para_ids:
                    placeholders = ','.join(['?' for _ in para_ids])
                    cursor.execute(f"SELECT COUNT(*) FROM similarity_results WHERE paragraph1_id NOT IN ({placeholders}) OR paragraph2_id NOT IN ({placeholders})", para_ids + para_ids)
                    orphaned_similarities = cursor.fetchone()[0]
                    print(f"Orphaned similarity results: {orphaned_similarities}")
        
        print("\nDiagnostic completed successfully")
        
    except Exception as e:
        print(f"ERROR during diagnostic: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

def reset_database(db_path=DEFAULT_DB_PATH):
    """Create a new database with proper schema, backing up the old one"""
    if os.path.exists(db_path):
        # Create backup
        backup_path = f"{db_path}.bak"
        print(f"Creating backup of existing database to {backup_path}")
        try:
            import shutil
            shutil.copy2(db_path, backup_path)
        except Exception as e:
            print(f"WARNING: Failed to create backup: {str(e)}")
    
    # Delete the existing database
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"Removed existing database: {db_path}")
        except Exception as e:
            print(f"ERROR: Could not remove existing database: {str(e)}")
            return False
    
    # Create a new database with proper schema
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Create Documents table
        cursor.execute('''
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            upload_date TEXT NOT NULL
        )
        ''')
        
        # Create Paragraphs table
        cursor.execute('''
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
        
        # Create Tags table
        cursor.execute('''
        CREATE TABLE tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            color TEXT NOT NULL
        )
        ''')
        
        # Create ParagraphTags table
        cursor.execute('''
        CREATE TABLE paragraph_tags (
            paragraph_id INTEGER NOT NULL,
            tag_id INTEGER NOT NULL,
            PRIMARY KEY (paragraph_id, tag_id),
            FOREIGN KEY (paragraph_id) REFERENCES paragraphs (id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id) REFERENCES tags (id) ON DELETE CASCADE
        )
        ''')
        
        # Create SimilarityResults table
        cursor.execute('''
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
        
        # Add indexes for performance
        cursor.execute('CREATE INDEX idx_paragraphs_document_id ON paragraphs (document_id)')
        cursor.execute('CREATE INDEX idx_similarity_paragraph1 ON similarity_results (paragraph1_id)')
        cursor.execute('CREATE INDEX idx_similarity_paragraph2 ON similarity_results (paragraph2_id)')
        
        # Create some default tags
        default_tags = [
            ('Header', '#007bff'),
            ('Footer', '#6c757d'),
            ('Disclaimer', '#dc3545'),
            ('Important', '#ffc107'),
            ('Common', '#28a745')
        ]
        
        for name, color in default_tags:
            cursor.execute('INSERT INTO tags (name, color) VALUES (?, ?)', (name, color))
        
        conn.commit()
        conn.close()
        
        print("New database created successfully with proper schema")
        return True
        
    except Exception as e:
        print(f"ERROR creating new database: {str(e)}")
        return False

def verify_orphaned_data(db_path=DEFAULT_DB_PATH):
    """Verify if there are any orphaned records and clean them"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Find orphaned paragraphs
        cursor.execute("""
            SELECT p.id FROM paragraphs p
            LEFT JOIN documents d ON p.document_id = d.id
            WHERE d.id IS NULL
        """)
        orphaned_paragraphs = cursor.fetchall()
        
        if orphaned_paragraphs:
            para_ids = [row[0] for row in orphaned_paragraphs]
            print(f"Found {len(para_ids)} orphaned paragraphs. IDs: {para_ids}")
            
            # Delete orphaned similarity results first
            placeholders = ','.join(['?' for _ in para_ids])
            cursor.execute(f"""
                DELETE FROM similarity_results
                WHERE paragraph1_id IN ({placeholders}) OR paragraph2_id IN ({placeholders})
            """, para_ids + para_ids)
            
            # Delete orphaned paragraph tags
            cursor.execute(f"""
                DELETE FROM paragraph_tags
                WHERE paragraph_id IN ({placeholders})
            """, para_ids)
            
            # Delete orphaned paragraphs
            cursor.execute(f"""
                DELETE FROM paragraphs
                WHERE id IN ({placeholders})
            """, para_ids)
            
            print(f"Deleted {len(para_ids)} orphaned paragraphs and related data")
            conn.commit()
        else:
            print("No orphaned paragraphs found")
        
        # Find orphaned similarity results
        cursor.execute("""
            SELECT s.id FROM similarity_results s
            LEFT JOIN paragraphs p1 ON s.paragraph1_id = p1.id
            LEFT JOIN paragraphs p2 ON s.paragraph2_id = p2.id
            WHERE p1.id IS NULL OR p2.id IS NULL
        """)
        orphaned_similarities = cursor.fetchall()
        
        if orphaned_similarities:
            sim_ids = [row[0] for row in orphaned_similarities]
            print(f"Found {len(sim_ids)} orphaned similarity results. IDs: {sim_ids}")
            
            placeholders = ','.join(['?' for _ in sim_ids])
            cursor.execute(f"""
                DELETE FROM similarity_results
                WHERE id IN ({placeholders})
            """, sim_ids)
            
            print(f"Deleted {len(sim_ids)} orphaned similarity results")
            conn.commit()
        else:
            print("No orphaned similarity results found")
        
        # Find orphaned paragraph tags
        cursor.execute("""
            SELECT pt.paragraph_id, pt.tag_id FROM paragraph_tags pt
            LEFT JOIN paragraphs p ON pt.paragraph_id = p.id
            LEFT JOIN tags t ON pt.tag_id = t.id
            WHERE p.id IS NULL OR t.id IS NULL
        """)
        orphaned_tags = cursor.fetchall()
        
        if orphaned_tags:
            print(f"Found {len(orphaned_tags)} orphaned paragraph tags")
            cursor.execute("""
                DELETE FROM paragraph_tags
                WHERE paragraph_id NOT IN (SELECT id FROM paragraphs)
                OR tag_id NOT IN (SELECT id FROM tags)
            """)
            print(f"Deleted orphaned paragraph tags")
            conn.commit()
        else:
            print("No orphaned paragraph tags found")
        
        conn.close()
        
    except Exception as e:
        print(f"ERROR verifying orphaned data: {str(e)}")
        if 'conn' in locals():
            conn.close()

def main():
    # Parse command line arguments
    db_path = DEFAULT_DB_PATH
    if len(sys.argv) > 1:
        if sys.argv[1] == "--reset":
            print("WARNING: This will reset the database and delete all existing data.")
            confirm = input("Are you sure you want to continue? (yes/no): ")
            if confirm.lower() == "yes":
                reset_database(db_path)
            else:
                print("Database reset cancelled")
            return
        elif sys.argv[1] == "--cleanup":
            print("Cleaning up orphaned data...")
            verify_orphaned_data(db_path)
            return
        else:
            db_path = sys.argv[1]
    
    # Run the diagnostic
    check_database(db_path)
    
    # Offer options
    print("\nOptions:")
    print("1. Reset database (this will delete all data)")
    print("2. Clean up orphaned data")
    print("3. Exit")
    
    choice = input("Enter your choice (1-3): ")
    if choice == "1":
        confirm = input("Are you sure you want to reset the database? All data will be lost. (yes/no): ")
        if confirm.lower() == "yes":
            reset_database(db_path)
    elif choice == "2":
        verify_orphaned_data(db_path)

if __name__ == "__main__":
    main()
