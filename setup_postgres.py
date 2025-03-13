#!/usr/bin/env python3
"""
PostgreSQL Setup Script for Paragraph Analyzer

This script helps set up the PostgreSQL database for the Paragraph Analyzer application.
It creates the database and necessary user if they don't exist.

Usage:
    python setup_postgres.py

Requirements:
    - PostgreSQL must be installed and running
    - psycopg2 package must be installed
    - User running this script must have PostgreSQL superuser privileges
"""

import getpass
import subprocess
import sys
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Database configuration
DB_NAME = "paragraph_analyzer"
DB_USER = "paragraph_user"
DB_PASSWORD = None  # Will prompt for password if None


def run_psql_command(command, admin_user=None, admin_password=None):
    """Run a PostgreSQL command using psql CLI."""
    cmd = ["psql"]
    
    if admin_user:
        cmd.extend(["-U", admin_user])
    
    cmd.extend(["-c", command])
    
    env = None
    if admin_password:
        env = {"PGPASSWORD": admin_password}
    
    try:
        result = subprocess.run(
            cmd, 
            env=env,
            capture_output=True, 
            text=True
        )
        return result.returncode == 0, result.stdout
    except Exception as e:
        print(f"Error running psql command: {e}")
        return False, str(e)


def create_database_and_user():
    """Create the database and user for the application."""
    print("\nPostgreSQL Setup for Paragraph Analyzer")
    print("=======================================")
    
    # Get admin credentials
    admin_user = input("\nEnter PostgreSQL admin username [postgres]: ") or "postgres"
    admin_password = getpass.getpass("Enter PostgreSQL admin password: ")
    
    # Get user password if not set
    global DB_PASSWORD
    if DB_PASSWORD is None:
        while True:
            DB_PASSWORD = getpass.getpass(f"\nEnter a password for the application user '{DB_USER}': ")
            confirm = getpass.getpass("Confirm password: ")
            if DB_PASSWORD == confirm:
                break
            print("Passwords don't match. Please try again.")
    
    # Connect to PostgreSQL with admin credentials
    try:
        print("\nConnecting to PostgreSQL...")
        conn = psycopg2.connect(
            user=admin_user,
            password=admin_password,
            host="localhost",
            port="5432"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Creating database '{DB_NAME}'...")
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
            print("Database created successfully.")
        else:
            print(f"Database '{DB_NAME}' already exists.")
        
        # Check if user exists
        cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (DB_USER,))
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Creating user '{DB_USER}'...")
            cursor.execute(
                sql.SQL("CREATE USER {} WITH ENCRYPTED PASSWORD %s").format(sql.Identifier(DB_USER)),
                (DB_PASSWORD,)
            )
            print("User created successfully.")
        else:
            print(f"User '{DB_USER}' already exists. Updating password...")
            cursor.execute(
                sql.SQL("ALTER USER {} WITH ENCRYPTED PASSWORD %s").format(sql.Identifier(DB_USER)),
                (DB_PASSWORD,)
            )
            print("User password updated.")
        
        # Grant privileges
        print(f"Granting privileges on '{DB_NAME}' to '{DB_USER}'...")
        cursor.execute(
            sql.SQL("GRANT ALL PRIVILEGES ON DATABASE {} TO {}").format(
                sql.Identifier(DB_NAME),
                sql.Identifier(DB_USER)
            )
        )
        
        # Connect to the new database to grant schema privileges
        cursor.close()
        conn.close()
        
        # Connect to the specific database
        conn = psycopg2.connect(
            user=admin_user,
            password=admin_password,
            host="localhost",
            port="5432",
            database=DB_NAME
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Grant schema privileges (for tables that will be created later)
        cursor.execute(
            sql.SQL("GRANT ALL ON SCHEMA public TO {}").format(
                sql.Identifier(DB_USER)
            )
        )
        print("Privileges granted successfully.")
        
        cursor.close()
        conn.close()
        
        print("\n✅ PostgreSQL setup completed successfully.")
        
        # Show connection string
        conn_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@localhost/{DB_NAME}"
        print("\nUpdate your app.py file with the following connection string:")
        print(f"DB_URL = \"{conn_string}\"")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error setting up PostgreSQL: {e}")
        return False


if __name__ == "__main__":
    if not create_database_and_user():
        sys.exit(1)
