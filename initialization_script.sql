-- Script to initialize or update the database with the file metadata table

-- Check if document_file_metadata table exists, if not, create it
CREATE TABLE IF NOT EXISTS document_file_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    file_size INTEGER,
    file_size_formatted TEXT,
    creation_date TEXT,
    modification_date TEXT,
    page_count INTEGER,
    paragraph_count INTEGER,
    image_count INTEGER,
    author TEXT,
    title TEXT,
    subject TEXT,
    creator TEXT,
    producer TEXT,
    pdf_version TEXT,
    is_encrypted BOOLEAN,
    has_signatures BOOLEAN,
    has_forms BOOLEAN,
    has_toc BOOLEAN,
    toc_items INTEGER,
    annotation_count INTEGER,
    fonts_used TEXT,  -- Stored as JSON array
    table_count INTEGER,
    section_count INTEGER,
    has_headers BOOLEAN,
    has_footers BOOLEAN,
    styles_used TEXT,  -- Stored as JSON array
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Create index for faster lookups if they don't exist already
CREATE INDEX IF NOT EXISTS idx_file_metadata_document_id ON document_file_metadata(document_id);
CREATE INDEX IF NOT EXISTS idx_file_metadata_page_count ON document_file_metadata(page_count);
CREATE INDEX IF NOT EXISTS idx_file_metadata_paragraph_count ON document_file_metadata(paragraph_count);
CREATE INDEX IF NOT EXISTS idx_file_metadata_file_size ON document_file_metadata(file_size);

-- For PostgreSQL, you might need to use:
/*
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'document_file_metadata'
    ) THEN
        CREATE TABLE document_file_metadata (
            id SERIAL PRIMARY KEY,
            document_id INTEGER NOT NULL,
            file_size BIGINT,
            file_size_formatted TEXT,
            creation_date TIMESTAMP,
            modification_date TIMESTAMP,
            page_count INTEGER,
            paragraph_count INTEGER,
            image_count INTEGER,
            author TEXT,
            title TEXT,
            subject TEXT,
            creator TEXT,
            producer TEXT,
            pdf_version TEXT,
            is_encrypted BOOLEAN,
            has_signatures BOOLEAN,
            has_forms BOOLEAN,
            has_toc BOOLEAN,
            toc_items INTEGER,
            annotation_count INTEGER,
            fonts_used TEXT,  -- Stored as JSON array
            table_count INTEGER,
            section_count INTEGER,
            has_headers BOOLEAN,
            has_footers BOOLEAN,
            styles_used TEXT,  -- Stored as JSON array
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        
        CREATE INDEX idx_file_metadata_document_id ON document_file_metadata(document_id);
        CREATE INDEX idx_file_metadata_page_count ON document_file_metadata(page_count);
        CREATE INDEX idx_file_metadata_paragraph_count ON document_file_metadata(paragraph_count);
        CREATE INDEX idx_file_metadata_file_size ON document_file_metadata(file_size);
    END IF;
END $;
*/

-- Execute this script with:
-- For SQLite: sqlite3 paragraph_analyzer.db < init_metadata_table.sql
-- For PostgreSQL: psql -U username -d paragraph_analyzer -f init_metadata_table.sql