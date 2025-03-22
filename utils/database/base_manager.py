import logging
from typing import TypeVar, Callable, Any, List, Dict

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .models import Base

# Type variable for generic session function
T = TypeVar('T')

class BaseManager:
    """
    Base manager class that handles database connection and session management.
    """
    def __init__(self, db_url: str, logging_level: str = 'INFO'):
        """
        Initialize the base manager.
        
        Args:
            db_url: Database connection URL
            logging_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.db_url = db_url
        self.logger = self._setup_logger(logging_level)
        
        # Create engine and session factory
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize database
        self._init_db()
    
    def _with_session(self, func: Callable[[Session], T]) -> T:
        """
        Execute a function with a database session, handling session lifecycle.
        
        Args:
            func: Function that takes a session as its argument
            
        Returns:
            The result of the provided function
        """
        session = self.Session()
        try:
            result = func(session)
            return result
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database error: {str(e)}", exc_info=True)
            raise
        finally:
            session.close()
    
    def _init_db(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        self.logger.info(f"Initializing database with SQLAlchemy using {self.db_url}")
        
        try:
            # Create tables if they don't exist
            Base.metadata.create_all(self.engine)
            self.logger.info("Database schema initialized")
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}", exc_info=True)
            raise
    
    def _setup_logger(self, level: str) -> logging.Logger:
        """
        Set up a logger instance.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        
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
        
        # Add console handler if not already added
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
