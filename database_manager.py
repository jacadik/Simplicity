import os
import logging
import sqlite3
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
from datetime import datetime
from document_parser import Paragraph
from similarity_analyzer import SimilarityResult


class DatabaseManager:
    """
    Manages all database operations for the paragraph analysis system.
    """
    def __init__(self, db_path: str, logging_level: str = 'INFO'):
        """Initialize the database manager."""
        self.db_path = db_path
        self.logger = self._setup_logger(logging_level)
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema if it doesn't exist."""
        self.logger.info(f"Initializing database at {self.db_path}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create Documents table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                upload_date TEXT NOT NULL
            )
            ''')
            
            # Create Paragraphs table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS paragraphs (
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
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                color TEXT NOT NULL
            )
            ''')
            
            # Create ParagraphTags table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS paragraph_tags (
                paragraph_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,
                PRIMARY KEY (paragraph_id, tag_id),
                FOREIGN KEY (paragraph_id) REFERENCES paragraphs (id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags (id) ON DELETE CASCADE
            )
            ''')
            
            # Create SimilarityResults table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS similarity_results (
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
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_paragraphs_document_id ON paragraphs (document_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_similarity_paragraph1 ON similarity_results (paragraph1_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_similarity_paragraph2 ON similarity_results (paragraph2_id)')
            
            # Create some default tags
            default_tags = [
                ('Header', '#007bff'),
                ('Footer', '#6c757d'),
                ('Disclaimer', '#dc3545'),
                ('Important', '#ffc107'),
                ('Common', '#28a745')
            ]
            
            for name, color in default_tags:
                cursor.execute('INSERT OR IGNORE INTO tags (name, color) VALUES (?, ?)', (name, color))
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}", exc_info=True)
            raise
    
    def add_document(self, filename: str, file_type: str, file_path: str) -> int:
        """Add a document to the database and return its ID."""
        self.logger.info(f"Adding document to database: {filename}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            upload_date = datetime.now().isoformat()
            
            cursor.execute(
                'INSERT INTO documents (filename, file_type, file_path, upload_date) VALUES (?, ?, ?, ?)',
                (filename, file_type, file_path, upload_date)
            )
            
            doc_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            self.logger.info(f"Document added with ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            self.logger.error(f"Error adding document: {str(e)}", exc_info=True)
            return -1
    
    def add_paragraphs(self, paragraphs: List[Paragraph]) -> List[int]:
        """Add paragraphs to the database and return their IDs."""
        if not paragraphs:
            return []
            
        self.logger.info(f"Adding {len(paragraphs)} paragraphs to database")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            paragraph_ids = []
            
            for para in paragraphs:
                cursor.execute(
                    '''INSERT INTO paragraphs 
                       (content, document_id, paragraph_type, position, header_content) 
                       VALUES (?, ?, ?, ?, ?)''',
                    (para.content, para.doc_id, para.paragraph_type, para.position, para.header_content)
                )
                paragraph_ids.append(cursor.lastrowid)
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Added {len(paragraph_ids)} paragraphs")
            return paragraph_ids
            
        except Exception as e:
            self.logger.error(f"Error adding paragraphs: {str(e)}", exc_info=True)
            return []
    
    def add_similarity_results(self, results: List[SimilarityResult]) -> bool:
        """Add similarity results to the database."""
        if not results:
            return True
            
        self.logger.info(f"Adding {len(results)} similarity results to database")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for result in results:
                cursor.execute(
                    '''INSERT INTO similarity_results 
                       (paragraph1_id, paragraph2_id, similarity_score, similarity_type) 
                       VALUES (?, ?, ?, ?)''',
                    (
                        result.paragraph1_id, 
                        result.paragraph2_id, 
                        result.similarity_score, 
                        result.similarity_type
                    )
                )
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Added {len(results)} similarity results")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding similarity results: {str(e)}", exc_info=True)
            return False
    
    def get_documents(self) -> List[Dict]:
        """Get all documents."""
        self.logger.info("Retrieving all documents")
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM documents ORDER BY upload_date DESC')
            rows = cursor.fetchall()
            
            documents = [dict(row) for row in rows]
            conn.close()
            
            self.logger.info(f"Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
            return []
    
    def get_paragraphs(self, document_id: Optional[int] = None, collapse_duplicates: bool = True) -> List[Dict]:
        """
        Get paragraphs, optionally filtered by document.
        If collapse_duplicates is True, exact matching paragraphs are displayed only once with document references.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if document_id is not None:
                self.logger.info(f"Retrieving paragraphs for document ID: {document_id}")
                cursor.execute(
                    '''SELECT p.*, d.filename 
                       FROM paragraphs p
                       JOIN documents d ON p.document_id = d.id 
                       WHERE p.document_id = ? 
                       ORDER BY p.position''',
                    (document_id,)
                )
            else:
                self.logger.info("Retrieving all paragraphs")
                cursor.execute(
                    '''SELECT p.*, d.filename 
                       FROM paragraphs p
                       JOIN documents d ON p.document_id = d.id 
                       ORDER BY p.document_id, p.position'''
                )
            
            rows = cursor.fetchall()
            
            # Get tags for each paragraph
            paragraphs = []
            for row in rows:
                para = dict(row)
                
                # Get tags for this paragraph
                cursor.execute(
                    '''SELECT t.id, t.name, t.color 
                       FROM tags t
                       JOIN paragraph_tags pt ON t.id = pt.tag_id
                       WHERE pt.paragraph_id = ?''',
                    (para['id'],)
                )
                tag_rows = cursor.fetchall()
                para['tags'] = [dict(tag) for tag in tag_rows]
                
                paragraphs.append(para)
            
            # Handle duplicate collapse if requested and not filtering by document
            if collapse_duplicates and document_id is None:
                # Get exact matching paragraphs based on content
                cursor.execute(
                    '''SELECT p1.content, GROUP_CONCAT(p1.id) as para_ids, GROUP_CONCAT(d.id) as doc_ids, 
                              GROUP_CONCAT(d.filename) as filenames
                       FROM paragraphs p1
                       JOIN documents d ON p1.document_id = d.id
                       GROUP BY p1.content
                       HAVING COUNT(*) > 1'''
                )
                duplicate_rows = cursor.fetchall()
                
                # Process duplicate paragraphs
                if duplicate_rows:
                    # Create a lookup of paragraph IDs that are duplicates
                    duplicate_ids = set()
                    duplicate_map = {}
                    
                    for dup in duplicate_rows:
                        content = dup['content']
                        para_ids = [int(pid) for pid in dup['para_ids'].split(',')]
                        doc_ids = [int(did) for did in dup['doc_ids'].split(',')]
                        filenames = dup['filenames'].split(',')
                        
                        # Keep track of all duplicate paragraph IDs (except the one we'll keep)
                        duplicate_ids.update(para_ids[1:])
                        
                        # Map the paragraph we'll keep to its document appearances
                        duplicate_map[para_ids[0]] = {
                            'doc_ids': doc_ids,
                            'filenames': filenames
                        }
                    
                    # Filter out duplicates and add document reference to kept paragraphs
                    filtered_paragraphs = []
                    for para in paragraphs:
                        if para['id'] in duplicate_ids:
                            # Skip this duplicate paragraph
                            continue
                        elif para['id'] in duplicate_map:
                            # This is the representative paragraph from a duplicate group
                            para['document_references'] = duplicate_map[para['id']]['filenames']
                            para['appears_in_multiple'] = True
                        else:
                            # Regular paragraph (no duplicates)
                            para['document_references'] = [para['filename']]
                            para['appears_in_multiple'] = False
                        
                        filtered_paragraphs.append(para)
                    
                    paragraphs = filtered_paragraphs
                else:
                    # No duplicates found, just add default document references
                    for para in paragraphs:
                        para['document_references'] = [para['filename']]
                        para['appears_in_multiple'] = False
            else:
                # Not collapsing duplicates, just add default document references
                for para in paragraphs:
                    para['document_references'] = [para['filename']]
                    para['appears_in_multiple'] = False
            
            conn.close()
            
            self.logger.info(f"Retrieved {len(paragraphs)} paragraphs")
            return paragraphs
            
        except Exception as e:
            self.logger.error(f"Error retrieving paragraphs: {str(e)}", exc_info=True)
            return []
    
    def get_similar_paragraphs(self, threshold: Optional[float] = None) -> List[Dict]:
        """Get similar paragraphs above the threshold."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = '''
                SELECT s.*, 
                       p1.content as para1_content, p1.document_id as para1_doc_id, d1.filename as para1_filename,
                       p2.content as para2_content, p2.document_id as para2_doc_id, d2.filename as para2_filename
                FROM similarity_results s
                JOIN paragraphs p1 ON s.paragraph1_id = p1.id
                JOIN paragraphs p2 ON s.paragraph2_id = p2.id
                JOIN documents d1 ON p1.document_id = d1.id
                JOIN documents d2 ON p2.document_id = d2.id
            '''
            
            if threshold is not None:
                self.logger.info(f"Retrieving similarity results with threshold: {threshold}")
                query += ' WHERE s.similarity_score >= ?'
                cursor.execute(query, (threshold,))
            else:
                self.logger.info("Retrieving all similarity results")
                cursor.execute(query)
            
            rows = cursor.fetchall()
            similarities = [dict(row) for row in rows]
            conn.close()
            
            self.logger.info(f"Retrieved {len(similarities)} similarity results")
            return similarities
            
        except Exception as e:
            self.logger.error(f"Error retrieving similarity results: {str(e)}", exc_info=True)
            return []
    
    def get_tags(self) -> List[Dict]:
        """Get all tags."""
        self.logger.info("Retrieving all tags")
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM tags')
            rows = cursor.fetchall()
            
            tags = [dict(row) for row in rows]
            conn.close()
            
            self.logger.info(f"Retrieved {len(tags)} tags")
            return tags
            
        except Exception as e:
            self.logger.error(f"Error retrieving tags: {str(e)}", exc_info=True)
            return []
    
    def add_tag(self, name: str, color: str) -> int:
        """Add a new tag and return its ID."""
        self.logger.info(f"Adding new tag: {name} with color {color}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('INSERT INTO tags (name, color) VALUES (?, ?)', (name, color))
            tag_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Added tag with ID: {tag_id}")
            return tag_id
            
        except Exception as e:
            self.logger.error(f"Error adding tag: {str(e)}", exc_info=True)
            return -1
    
    def tag_paragraph(self, paragraph_id: int, tag_id: int) -> bool:
        """Associate a tag with a paragraph."""
        self.logger.info(f"Tagging paragraph {paragraph_id} with tag {tag_id}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert or ignore in case the association already exists
            cursor.execute(
                'INSERT OR IGNORE INTO paragraph_tags (paragraph_id, tag_id) VALUES (?, ?)',
                (paragraph_id, tag_id)
            )
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Tagged paragraph {paragraph_id} with tag {tag_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error tagging paragraph: {str(e)}", exc_info=True)
            return False
    
    def untag_paragraph(self, paragraph_id: int, tag_id: int) -> bool:
        """Remove a tag association from a paragraph."""
        self.logger.info(f"Removing tag {tag_id} from paragraph {paragraph_id}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'DELETE FROM paragraph_tags WHERE paragraph_id = ? AND tag_id = ?',
                (paragraph_id, tag_id)
            )
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Removed tag {tag_id} from paragraph {paragraph_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing tag from paragraph: {str(e)}", exc_info=True)
            return False
    
    def export_to_excel(self, output_path: str) -> bool:
        """Export database contents to Excel."""
        self.logger.info(f"Exporting data to Excel: {output_path}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get all paragraphs with document info
            paragraphs_df = pd.read_sql_query(
                '''SELECT p.id, p.content, p.paragraph_type, p.header_content, 
                          d.filename, d.upload_date
                   FROM paragraphs p
                   JOIN documents d ON p.document_id = d.id
                   ORDER BY d.filename, p.position''',
                conn
            )
            
            # Get all tags for each paragraph
            paragraph_tags = pd.read_sql_query(
                '''SELECT pt.paragraph_id, GROUP_CONCAT(t.name, ', ') as tags
                   FROM paragraph_tags pt
                   JOIN tags t ON pt.tag_id = t.id
                   GROUP BY pt.paragraph_id''',
                conn
            )
            
            # Merge paragraphs with their tags
            if not paragraph_tags.empty:
                paragraphs_df = pd.merge(
                    paragraphs_df, 
                    paragraph_tags, 
                    left_on='id', 
                    right_on='paragraph_id', 
                    how='left'
                )
            else:
                paragraphs_df['tags'] = None
            
            # Get similarity data
            similarity_df = pd.read_sql_query(
                '''SELECT s.similarity_score, s.similarity_type,
                          p1.content as paragraph1_content, d1.filename as document1,
                          p2.content as paragraph2_content, d2.filename as document2
                   FROM similarity_results s
                   JOIN paragraphs p1 ON s.paragraph1_id = p1.id
                   JOIN paragraphs p2 ON s.paragraph2_id = p2.id
                   JOIN documents d1 ON p1.document_id = d1.id
                   JOIN documents d2 ON p2.document_id = d2.id
                   WHERE s.similarity_score >= 0.8
                   ORDER BY s.similarity_score DESC''',
                conn
            )
            
            conn.close()
            
            # Write to Excel file
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                # Paragraphs sheet
                paragraphs_df.to_excel(writer, sheet_name='Paragraphs', index=False)
                
                # Similarities sheet
                if not similarity_df.empty:
                    similarity_df.to_excel(writer, sheet_name='Similarities', index=False)
                
                # Format the sheets
                workbook = writer.book
                
                # Format for the Paragraphs sheet
                sheet1 = writer.sheets['Paragraphs']
                sheet1.set_column('A:A', 10)  # ID column
                sheet1.set_column('B:B', 80)  # Content column
                sheet1.set_column('C:E', 20)  # Other columns
                
                # Format for the Similarities sheet if it exists
                if not similarity_df.empty:
                    sheet2 = writer.sheets['Similarities']
                    sheet2.set_column('A:A', 15)  # Score column
                    sheet2.set_column('B:B', 15)  # Type column
                    sheet2.set_column('C:D', 80)  # Content columns
                    sheet2.set_column('E:F', 30)  # Document columns
            
            self.logger.info(f"Data exported successfully to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting to Excel: {str(e)}", exc_info=True)
            return False
    
    def delete_document(self, document_id: int) -> bool:
        """Delete a document and its paragraphs."""
        self.logger.info(f"Deleting document with ID: {document_id}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get the file path to delete the file as well
            cursor.execute('SELECT file_path FROM documents WHERE id = ?', (document_id,))
            result = cursor.fetchone()
            
            if result:
                file_path = result[0]
                
                # Delete from database (cascade will handle related records)
                cursor.execute('DELETE FROM documents WHERE id = ?', (document_id,))
                conn.commit()
                
                # Try to delete the file if it exists
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.logger.info(f"Deleted file: {file_path}")
                
                self.logger.info(f"Deleted document with ID: {document_id}")
                return True
            else:
                self.logger.warning(f"Document with ID {document_id} not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Error deleting document: {str(e)}", exc_info=True)
            return False
        finally:
            if 'conn' in locals():
                conn.close()
    
    def clear_database(self) -> bool:
        """Clear all data from the database."""
        self.logger.info("Clearing all data from database")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all file paths to delete files as well
            cursor.execute('SELECT file_path FROM documents')
            file_paths = [row[0] for row in cursor.fetchall()]
            
            # Delete all records
            cursor.execute('DELETE FROM similarity_results')
            cursor.execute('DELETE FROM paragraph_tags')
            cursor.execute('DELETE FROM paragraphs')
            cursor.execute('DELETE FROM documents')
            # Don't delete tags
            
            conn.commit()
            
            # Delete all files
            for file_path in file_paths:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.logger.info(f"Deleted file: {file_path}")
            
            self.logger.info("Database cleared successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing database: {str(e)}", exc_info=True)
            return False
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _setup_logger(self, level: str) -> logging.Logger:
        """Set up a logger instance."""
        logger = logging.getLogger(f'{__name__}.DatabaseManager')
        
        if level.upper() == 'DEBUG':
            logger.setLevel(logging.DEBUG)
        elif level.upper() == 'INFO':
            logger.setLevel(logging.INFO)
        elif level.upper() == 'WARNING':
            logger.setLevel(logging.WARNING)
        elif level.upper() == 'ERROR':
            logger.setLevel(logging.ERROR)
        else:
            logger.setLevel(logging.INFO)
        
        return logger
