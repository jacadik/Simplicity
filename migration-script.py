#!/usr/bin/env python3
"""
PostgreSQL Migration Script for Dual Similarity Metrics

This script updates the similarity_results table to add text_similarity_score
and rename similarity_score to content_similarity_score.

Usage:
    python migrate_postgres.py [--db-url postgresql://username:password@localhost/paragraph_analyzer]
"""

import argparse
import logging
import sys
from datetime import datetime
from sqlalchemy import create_engine, text, inspect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"migration_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("postgres_migration")

# Default PostgreSQL connection URL
DEFAULT_DB_URL = "postgresql://paragraph_user:pass@localhost/paragraph_analyzer"

def run_migration(db_url):
    """
    Run the migration on PostgreSQL database
    """
    logger.info(f"Starting migration on database: {db_url}")
    
    try:
        # Create SQLAlchemy engine with echo for debugging
        engine = create_engine(db_url, echo=False)
        connection = engine.connect()
        
        # Begin a transaction
        trans = connection.begin()
        try:
            # Check if table exists
            inspector = inspect(engine)
            if 'similarity_results' not in inspector.get_table_names():
                logger.error("The similarity_results table does not exist")
                trans.rollback()
                return False
            
            # Check columns
            columns = [col['name'] for col in inspector.get_columns('similarity_results')]
            logger.info(f"Current columns: {', '.join(columns)}")
            
            # Create a backup if supported by your PostgreSQL setup
            # (Uncomment if you have proper permissions and want a backup)
            # connection.execute(text("CREATE TABLE similarity_results_backup AS SELECT * FROM similarity_results"))
            # logger.info("Created backup table: similarity_results_backup")
            
            # 1. Add text_similarity_score column if it doesn't exist
            if 'text_similarity_score' not in columns:
                logger.info("Adding text_similarity_score column...")
                connection.execute(text("ALTER TABLE similarity_results ADD COLUMN text_similarity_score FLOAT"))
                
                # Initialize with the same value as similarity_score
                if 'similarity_score' in columns:
                    connection.execute(text("UPDATE similarity_results SET text_similarity_score = similarity_score"))
                elif 'content_similarity_score' in columns:
                    connection.execute(text("UPDATE similarity_results SET text_similarity_score = content_similarity_score"))
                
                logger.info("Added and initialized text_similarity_score column")
            else:
                logger.info("Column text_similarity_score already exists, skipping addition")
            
            # 2. Rename similarity_score to content_similarity_score if needed
            if 'similarity_score' in columns and 'content_similarity_score' not in columns:
                logger.info("Renaming similarity_score to content_similarity_score...")
                connection.execute(text("ALTER TABLE similarity_results RENAME COLUMN similarity_score TO content_similarity_score"))
                logger.info("Renamed similarity_score to content_similarity_score")
            elif 'content_similarity_score' in columns:
                logger.info("Column content_similarity_score already exists, skipping rename")
            
            # 3. Ensure indices exist for performance
            # Check existing indices
            indices = [idx['name'] for idx in inspector.get_indexes('similarity_results')]
            logger.info(f"Current indices: {', '.join(indices) if indices else 'None'}")
            
            # Create content_similarity index if needed
            if 'idx_content_similarity' not in indices:
                logger.info("Creating index on content_similarity_score...")
                connection.execute(text("CREATE INDEX idx_content_similarity ON similarity_results (content_similarity_score)"))
            
            # Create text_similarity index if needed
            if 'idx_text_similarity' not in indices:
                logger.info("Creating index on text_similarity_score...")
                connection.execute(text("CREATE INDEX idx_text_similarity ON similarity_results (text_similarity_score)"))
            
            # Commit the transaction
            trans.commit()
            logger.info("Migration completed successfully")
            
            # Verify the changes
            inspector = inspect(engine)
            columns = [col['name'] for col in inspector.get_columns('similarity_results')]
            logger.info(f"Final columns: {', '.join(columns)}")
            
            return True
            
        except Exception as e:
            # Rollback the transaction on error
            trans.rollback()
            logger.error(f"Transaction rolled back due to error: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        return False
    finally:
        if 'connection' in locals():
            connection.close()

def validate_migration(db_url):
    """
    Validate the migration was successful
    """
    logger.info(f"Validating migration on database: {db_url}")
    
    try:
        # Create SQLAlchemy engine
        engine = create_engine(db_url)
        connection = engine.connect()
        
        # Check columns
        inspector = inspect(engine)
        columns = [col['name'] for col in inspector.get_columns('similarity_results')]
        
        # Verify required columns exist
        required_columns = ['content_similarity_score', 'text_similarity_score']
        missing_columns = [col for col in required_columns if col not in columns]
        
        if missing_columns:
            logger.error(f"Validation failed: Missing columns: {', '.join(missing_columns)}")
            return False
        
        # Check for sample data
        result = connection.execute(text("SELECT COUNT(*) FROM similarity_results"))
        count = result.scalar()
        logger.info(f"Found {count} records in similarity_results table")
        
        if count > 0:
            # Check sample data
            result = connection.execute(text("""
                SELECT id, paragraph1_id, paragraph2_id, 
                       content_similarity_score, text_similarity_score, similarity_type 
                FROM similarity_results LIMIT 3
            """))
            rows = result.fetchall()
            
            # Log sample data
            logger.info("Sample data after migration:")
            for row in rows:
                logger.info(f"  {row}")
        
        logger.info("Migration validation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return False
    finally:
        if 'connection' in locals():
            connection.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Migrate PostgreSQL database for dual similarity metrics')
    parser.add_argument('--db-url', default=DEFAULT_DB_URL, 
                        help='PostgreSQL connection URL (default: %(default)s)')
    args = parser.parse_args()
    
    db_url = args.db_url
    
    print("\n==== PostgreSQL Migration for Dual Similarity Metrics ====")
    print(f"Database URL: {db_url}")
    confirm = input("Continue with migration? (yes/no): ")
    
    if confirm.lower() != "yes":
        print("Migration aborted by user")
        sys.exit(0)
    
    if run_migration(db_url):
        print("\nMigration successful!")
        print("\nRunning validation...")
        validate_migration(db_url)
    else:
        print("\nMigration failed. Check the log file for details.")
        sys.exit(1)