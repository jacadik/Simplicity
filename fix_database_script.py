#!/usr/bin/env python
"""
Improved Database Schema Fix Script
----------------------------------
This script adds the missing file_type column to the inserts table
and handles multiple database types (PostgreSQL, SQLite).

Usage:
    python improved_fix_database.py
"""

import os
import sys
import logging
import traceback
from sqlalchemy import create_engine, text, inspect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('db_fix')

# Get database URL from app.py or environment if possible,
# otherwise use this fallback
DB_URL = "postgresql://paragraph_user:pass@localhost/paragraph_analyzer"

# Try to import from app.py to get database URL
try:
    # Add project directory to path if needed
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from app import DB_URL as APP_DB_URL
    DB_URL = APP_DB_URL
    logger.info(f"Using database URL from app.py: {DB_URL}")
except Exception:
    logger.info(f"Using default database URL: {DB_URL}")


def fix_database():
    """Fix the database schema by adding the missing file_type column."""
    logger.info("Starting database schema fix")
    
    try:
        # Create engine
        engine = create_engine(DB_URL)
        logger.info(f"Connected to database")
        
        # Determine database type
        is_sqlite = DB_URL.startswith('sqlite')
        is_postgresql = 'postgresql' in DB_URL
        
        with engine.connect() as connection:
            # Check if inserts table exists
            inspector = inspect(engine)
            table_exists = 'inserts' in inspector.get_table_names()
            
            if not table_exists:
                logger.error("The 'inserts' table does not exist. Run your application first to create it.")
                return False
            
            # Get columns in the inserts table
            columns = [col['name'] for col in inspector.get_columns('inserts')]
            column_exists = 'file_type' in columns
            
            if column_exists:
                logger.info("The file_type column already exists. No fix needed.")
                return True
            
            logger.info("The file_type column is missing. Will add it now.")
            
            # Begin transaction
            trans = connection.begin()
            try:
                # Add file_type column (different syntax based on database type)
                if is_sqlite:
                    logger.info("Adding file_type column to SQLite database...")
                    # SQLite has limited ALTER TABLE support - we can add but not set in one statement
                    connection.execute(text("ALTER TABLE inserts ADD COLUMN file_type TEXT"))
                    
                    # Update existing records in SQLite
                    connection.execute(text("""
                        UPDATE inserts 
                        SET file_type = LOWER(
                            SUBSTR(filename, 
                                INSTR(filename, '.') + 1, 
                                LENGTH(filename)
                            )
                        )
                        WHERE INSTR(filename, '.') > 0
                    """))
                    
                    # Set 'unknown' for records without an extension
                    connection.execute(text("""
                        UPDATE inserts 
                        SET file_type = 'unknown'
                        WHERE file_type IS NULL OR file_type = ''
                    """))
                    
                elif is_postgresql:
                    logger.info("Adding file_type column to PostgreSQL database...")
                    connection.execute(text("ALTER TABLE inserts ADD COLUMN file_type VARCHAR"))
                    
                    # Update PostgreSQL records
                    connection.execute(text("""
                        UPDATE inserts 
                        SET file_type = LOWER(
                            CASE 
                                WHEN position('.' in filename) > 0 
                                THEN split_part(filename, '.', -1)
                                ELSE 'unknown' 
                            END
                        )
                    """))
                else:
                    logger.info("Adding file_type column to database...")
                    # Generic approach for other databases
                    connection.execute(text("ALTER TABLE inserts ADD COLUMN file_type VARCHAR"))
                    
                    # We can't reliably update the values for unknown databases,
                    # but at least the column is added
                    logger.warning(
                        "Added file_type column, but couldn't automatically update values. "
                        "You may need to manually update the file_type values."
                    )
                
                # Commit transaction
                trans.commit()
                logger.info("Database schema updated successfully!")
                return True
                
            except Exception as e:
                # Rollback on error
                trans.rollback()
                logger.error(f"Error updating schema: {str(e)}")
                logger.error(traceback.format_exc())
                return False
        
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    print("\n=== PARAGRAPH ANALYZER DATABASE FIX SCRIPT ===\n")
    
    try:
        success = fix_database()
        if success:
            print("\n✅ Database fix completed successfully.\n")
            sys.exit(0)
        else:
            print("\n❌ Database fix failed. Check the error messages above.\n")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)