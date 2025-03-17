#!/usr/bin/env python3
"""
Database Migration Script for Paragraph Analyzer

This script adds the 'column' field to the paragraphs table.
Run this script before using the updated document parser.

Usage:
    python migrate_database.py [--db-url DB_URL]
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"migration_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("database_migration")

def migrate_database(db_url):
    """
    Add 'column' field to paragraphs table if it doesn't exist.
    
    Args:
        db_url: Database connection URL
        
    Returns:
        bool: Success status
    """
    try:
        # Import SQLAlchemy components
        from sqlalchemy import create_engine, inspect, text
        
        logger.info(f"Starting migration on database: {db_url}")
        
        # Create engine and connect
        engine = create_engine(db_url)
        inspector = inspect(engine)
        
        # Check if paragraphs table exists
        if 'paragraphs' not in inspector.get_table_names():
            logger.error("Paragraphs table does not exist in the database")
            return False
        
        # Check if column already exists
        columns = [col['name'] for col in inspector.get_columns('paragraphs')]
        logger.info(f"Current columns in paragraphs table: {', '.join(columns)}")
        
        if 'column' in columns:
            logger.info("Column 'column' already exists in paragraphs table. No migration needed.")
            return True
        
        # Add the column based on database type
        with engine.connect() as conn:
            if 'sqlite' in db_url.lower():
                # SQLite syntax
                conn.execute(text("ALTER TABLE paragraphs ADD COLUMN column INTEGER"))
                logger.info("Added 'column' field to paragraphs table (SQLite)")
            else:
                # PostgreSQL syntax
                conn.execute(text('ALTER TABLE paragraphs ADD COLUMN "column" INTEGER'))
                logger.info("Added 'column' field to paragraphs table (PostgreSQL)")
            
            # Commit the transaction
            conn.commit()
        
        # Verify the column was added
        inspector = inspect(engine)
        columns_after = [col['name'] for col in inspector.get_columns('paragraphs')]
        
        if 'column' in columns_after:
            logger.info("Migration completed successfully")
            return True
        else:
            logger.error("Migration failed: Column not added")
            return False
            
    except ImportError as e:
        logger.error(f"Required package not installed: {str(e)}")
        logger.error("Please install SQLAlchemy: pip install sqlalchemy")
        return False
    except Exception as e:
        logger.error(f"Error during migration: {str(e)}")
        return False

def main():
    """Main entry point for the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Migrate Paragraph Analyzer database')
    
    # Default to PostgreSQL URL, can be overridden with command line arg
    default_db_url = "postgresql://paragraph_user:pass@localhost/paragraph_analyzer"
    
    parser.add_argument('--db-url', default=default_db_url,
                        help=f'Database URL (default: {default_db_url})')
    
    args = parser.parse_args()
    
    # Run migration
    if migrate_database(args.db_url):
        print("\nDatabase migration completed successfully.")
        return 0
    else:
        print("\nDatabase migration failed. Check the log for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
