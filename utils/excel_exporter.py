#!/usr/bin/env python3
"""
Excel Exporter for Paragraph Analyzer

This standalone module exports paragraph analysis data to Excel format
with comprehensive sheets for business analysis.
"""

import os
import sys
import argparse
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, or_, func
from sqlalchemy.orm import sessionmaker

# Import database models
from utils.database.models import (
    Document, Paragraph, Tag, SimilarityResult, Cluster,
    paragraph_tags, cluster_paragraphs
)

# Configure logging
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'excel_export.log')

def setup_logging(level='INFO'):
    """Set up logging configuration."""
    logger = logging.getLogger('excel_exporter')
    
    # Set log level
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
    
    # Create handlers if they don't exist
    if not logger.handlers:
        # File handler with rotation
        file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10485760, backupCount=5)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

def export_to_excel(db_url, output_path, logger):
    """
    Export database contents to Excel with comprehensive data for business analysis.
    
    Args:
        db_url: Database connection URL
        output_path: Path to save the Excel file
        logger: Logger instance
    
    Returns:
        bool: Success status
    """
    logger.info(f"Exporting data to Excel: {output_path}")
    
    try:
        # Create SQLAlchemy engine and session
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # --- 1. Paragraphs Data ---
        logger.info("Collecting paragraphs data")
        paragraphs_data = []
        
        # Query paragraphs with document info
        results = session.query(
            Paragraph, 
            Document.filename, 
            Document.upload_date,
            Document.file_type
        ).join(
            Document
        ).order_by(
            Document.filename, 
            Paragraph.position
        ).all()
        
        # Process paragraphs and include tags and word/character counts
        for para, filename, upload_date, file_type in results:
            # Get tags as comma-separated string
            tags_str = ', '.join([tag.name for tag in para.tags]) if para.tags else ''
            
            # Calculate word and character counts
            word_count = len(para.content.split()) if para.content else 0
            char_count = len(para.content) if para.content else 0
            
            # Get clusters this paragraph belongs to
            clusters_str = ', '.join([cluster.name for cluster in para.clusters]) if para.clusters else ''
            
            paragraphs_data.append({
                'Paragraph ID': para.id,
                'Document': filename,
                'Document Type': file_type,
                'Upload Date': upload_date,
                'Position in Document': para.position,
                'Paragraph Type': para.paragraph_type,
                'Word Count': word_count,
                'Character Count': char_count,
                'Header Content': para.header_content if para.header_content else '',
                'Content': para.content,
                'Tags': tags_str,
                'Clusters': clusters_str
            })
        
        # Convert to DataFrame
        paragraphs_df = pd.DataFrame(paragraphs_data)
        
        # --- 2. Similarity Data ---
        logger.info("Collecting similarity data")
        similarity_data = []
        
        # Query all similarity results
        similarity_results = session.query(
            SimilarityResult,
            Paragraph.content.label('paragraph1_content'),
            Document.filename.label('document1')
        ).join(
            Paragraph,
            SimilarityResult.paragraph1_id == Paragraph.id
        ).join(
            Document,
            Paragraph.document_id == Document.id
        ).all()
        
        # Get paragraph2 and document2 data
        for sim, para1_content, doc1_filename in similarity_results:
            para1 = session.query(Paragraph).get(sim.paragraph1_id)
            para2 = session.query(Paragraph).get(sim.paragraph2_id)
            doc1 = session.query(Document).get(para1.document_id)
            doc2 = session.query(Document).get(para2.document_id)
            
            # Calculate additional metrics
            para1_words = para1.content.split() if para1.content else []
            para2_words = para2.content.split() if para2.content else []
            word_diff = abs(len(para1_words) - len(para2_words))
            char_diff = abs(len(para1.content) - len(para2.content)) if para1.content and para2.content else 0
            
            similarity_data.append({
                'Similarity ID': sim.id,
                'Paragraph 1 ID': sim.paragraph1_id,
                'Paragraph 2 ID': sim.paragraph2_id,
                'Content Similarity (%)': round(sim.content_similarity_score * 100, 2),
                'Text Similarity (%)': round(sim.text_similarity_score * 100, 2) if sim.text_similarity_score is not None else None,
                'Similarity Type': sim.similarity_type,
                'Document 1': doc1_filename,
                'Document 2': doc2.filename,
                'Word Count Difference': word_diff,
                'Character Count Difference': char_diff,
                'Paragraph 1 Type': para1.paragraph_type,
                'Paragraph 2 Type': para2.paragraph_type,
                'Paragraph 1 Content': para1.content,
                'Paragraph 2 Content': para2.content
            })
        
        # Convert to DataFrame
        similarity_df = pd.DataFrame(similarity_data)
        
        # --- 3. Document Statistics ---
        logger.info("Collecting document statistics")
        document_data = []
        
        # Query all documents with paragraph counts
        documents = session.query(Document).all()
        
        for doc in documents:
            # Count paragraphs by type
            para_counts = {}
            total_paragraphs = 0
            total_words = 0
            total_chars = 0
            
            for para in doc.paragraphs:
                para_type = para.paragraph_type
                if para_type not in para_counts:
                    para_counts[para_type] = 0
                para_counts[para_type] += 1
                total_paragraphs += 1
                
                # Count words and characters
                word_count = len(para.content.split()) if para.content else 0
                char_count = len(para.content) if para.content else 0
                total_words += word_count
                total_chars += char_count
            
            # Get similarity stats
            similarity_count = session.query(SimilarityResult).filter(
                or_(
                    SimilarityResult.paragraph1_id.in_([p.id for p in doc.paragraphs]),
                    SimilarityResult.paragraph2_id.in_([p.id for p in doc.paragraphs])
                )
            ).count()
            
            # Format paragraph type counts
            para_type_str = '; '.join([f"{k}: {v}" for k, v in para_counts.items()])
            
            document_data.append({
                'Document ID': doc.id,
                'Filename': doc.filename,
                'File Type': doc.file_type,
                'Upload Date': doc.upload_date,
                'Total Paragraphs': total_paragraphs,
                'Total Words': total_words,
                'Total Characters': total_chars,
                'Paragraph Types': para_type_str,
                'Similarity Connections': similarity_count
            })
        
        # Convert to DataFrame
        documents_df = pd.DataFrame(document_data)
        
        # --- 4. Tag Statistics ---
        logger.info("Collecting tag statistics")
        tag_data = []
        
        # Query all tags with counts
        tags = session.query(Tag).all()
        
        for tag in tags:
            tag_data.append({
                'Tag ID': tag.id,
                'Tag Name': tag.name,
                'Tag Color': tag.color,
                'Paragraphs Count': len(tag.paragraphs),
                'Documents Count': len(set(para.document_id for para in tag.paragraphs)) if tag.paragraphs else 0
            })
        
        # Convert to DataFrame
        tags_df = pd.DataFrame(tag_data)
        
        # --- 5. Cluster Statistics ---
        logger.info("Collecting cluster statistics")
        cluster_data = []
        
        # Query all clusters
        clusters = session.query(Cluster).all()
        
        for cluster in clusters:
            # Get document distribution
            doc_ids = set(para.document_id for para in cluster.paragraphs)
            doc_names = [session.query(Document.filename).filter_by(id=doc_id).scalar() for doc_id in doc_ids]
            
            # Get paragraph types distribution
            para_types = {}
            for para in cluster.paragraphs:
                if para.paragraph_type not in para_types:
                    para_types[para.paragraph_type] = 0
                para_types[para.paragraph_type] += 1
            
            para_types_str = '; '.join([f"{k}: {v}" for k, v in para_types.items()])
            
            cluster_data.append({
                'Cluster ID': cluster.id,
                'Cluster Name': cluster.name,
                'Creation Date': cluster.creation_date,
                'Similarity Threshold': cluster.similarity_threshold,
                'Similarity Type': cluster.similarity_type,
                'Paragraph Count': len(cluster.paragraphs),
                'Document Count': len(doc_ids),
                'Documents': ', '.join(doc_names),
                'Paragraph Types': para_types_str
            })
        
        # Convert to DataFrame
        clusters_df = pd.DataFrame(cluster_data)
        
        # --- 6. Summary Statistics ---
        logger.info("Collecting summary statistics")
        summary_data = []
        
        # Count totals
        total_documents = session.query(func.count(Document.id)).scalar()
        total_paragraphs = session.query(func.count(Paragraph.id)).scalar()
        total_similarities = session.query(func.count(SimilarityResult.id)).scalar()
        total_exact_matches = session.query(func.count(SimilarityResult.id)).filter_by(similarity_type='exact').scalar()
        total_similar_matches = session.query(func.count(SimilarityResult.id)).filter_by(similarity_type='similar').scalar()
        total_tags = session.query(func.count(Tag.id)).scalar()
        total_clusters = session.query(func.count(Cluster.id)).scalar()
        
        # Get average similarity scores
        avg_content_similarity = session.query(func.avg(SimilarityResult.content_similarity_score)).scalar()
        avg_text_similarity = session.query(func.avg(SimilarityResult.text_similarity_score)).scalar()
        
        # Export timestamp
        export_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create summary entries
        summary_data.append({'Metric': 'Export Date', 'Value': export_time})
        summary_data.append({'Metric': 'Total Documents', 'Value': total_documents})
        summary_data.append({'Metric': 'Total Paragraphs', 'Value': total_paragraphs})
        summary_data.append({'Metric': 'Total Similarity Matches', 'Value': total_similarities})
        summary_data.append({'Metric': 'Exact Matches', 'Value': total_exact_matches})
        summary_data.append({'Metric': 'Similar Matches', 'Value': total_similar_matches})
        summary_data.append({
            'Metric': 'Average Content Similarity',
            'Value': f"{avg_content_similarity*100:.2f}%" if avg_content_similarity is not None else 'N/A'
        })
        summary_data.append({
            'Metric': 'Average Text Similarity',
            'Value': f"{avg_text_similarity*100:.2f}%" if avg_text_similarity is not None else 'N/A'
        })
        summary_data.append({'Metric': 'Total Tags', 'Value': total_tags})
        summary_data.append({'Metric': 'Total Clusters', 'Value': total_clusters})
        
        # Convert to DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Close session
        session.close()
        
        # --- Write Excel File ---
        logger.info("Writing Excel file")
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # Write all sheets
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            documents_df.to_excel(writer, sheet_name='Documents', index=False)
            paragraphs_df.to_excel(writer, sheet_name='Paragraphs', index=False)
            
            if not similarity_df.empty:
                similarity_df.to_excel(writer, sheet_name='Similarities', index=False)
            
            if not tags_df.empty:
                tags_df.to_excel(writer, sheet_name='Tags', index=False)
            
            if not clusters_df.empty:
                clusters_df.to_excel(writer, sheet_name='Clusters', index=False)
            
            # Get the workbook
            workbook = writer.book
            
            # Create header format
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            # Format Summary sheet
            logger.debug("Formatting Summary sheet")
            sheet_summary = writer.sheets['Summary']
            for col_num, value in enumerate(summary_df.columns.values):
                sheet_summary.write(0, col_num, value, header_format)
            sheet_summary.set_column('A:A', 30)  # Metric column
            sheet_summary.set_column('B:B', 20)  # Value column
            
            # Format Documents sheet
            logger.debug("Formatting Documents sheet")
            sheet_docs = writer.sheets['Documents']
            for col_num, value in enumerate(documents_df.columns.values):
                sheet_docs.write(0, col_num, value, header_format)
            sheet_docs.set_column('A:A', 10)  # ID column
            sheet_docs.set_column('B:B', 30)  # Filename column
            sheet_docs.set_column('C:E', 15)  # Type, date, count columns
            sheet_docs.set_column('F:H', 12)  # Counts columns
            sheet_docs.set_column('I:I', 40)  # Paragraph types column
            sheet_docs.set_column('J:J', 15)  # Similarity column
            
            # Format Paragraphs sheet
            logger.debug("Formatting Paragraphs sheet")
            sheet_paras = writer.sheets['Paragraphs']
            for col_num, value in enumerate(paragraphs_df.columns.values):
                sheet_paras.write(0, col_num, value, header_format)
            sheet_paras.set_column('A:A', 12)  # ID column
            sheet_paras.set_column('B:C', 25)  # Document columns
            sheet_paras.set_column('D:D', 20)  # Upload date
            sheet_paras.set_column('E:F', 15)  # Position and type
            sheet_paras.set_column('G:H', 12)  # Count columns
            sheet_paras.set_column('I:I', 30)  # Header content
            sheet_paras.set_column('J:J', 80)  # Content column
            sheet_paras.set_column('K:L', 30)  # Tags and clusters
            
            # Format Similarities sheet if not empty
            if not similarity_df.empty:
                logger.debug("Formatting Similarities sheet")
                sheet_sim = writer.sheets['Similarities']
                for col_num, value in enumerate(similarity_df.columns.values):
                    sheet_sim.write(0, col_num, value, header_format)
                sheet_sim.set_column('A:A', 12)  # ID column
                sheet_sim.set_column('B:C', 15)  # Paragraph IDs
                sheet_sim.set_column('D:E', 15)  # Similarity scores
                sheet_sim.set_column('F:F', 15)  # Type
                sheet_sim.set_column('G:H', 25)  # Document names
                sheet_sim.set_column('I:J', 15)  # Difference metrics
                sheet_sim.set_column('K:L', 15)  # Paragraph types
                sheet_sim.set_column('M:N', 80)  # Content columns
            
            # Format Tags sheet if not empty
            if not tags_df.empty:
                logger.debug("Formatting Tags sheet")
                sheet_tags = writer.sheets['Tags']
                for col_num, value in enumerate(tags_df.columns.values):
                    sheet_tags.write(0, col_num, value, header_format)
                sheet_tags.set_column('A:A', 10)  # ID column
                sheet_tags.set_column('B:B', 25)  # Name column
                sheet_tags.set_column('C:C', 15)  # Color column
                sheet_tags.set_column('D:E', 15)  # Count columns
            
            # Format Clusters sheet if not empty
            if not clusters_df.empty:
                logger.debug("Formatting Clusters sheet")
                sheet_clusters = writer.sheets['Clusters']
                for col_num, value in enumerate(clusters_df.columns.values):
                    sheet_clusters.write(0, col_num, value, header_format)
                sheet_clusters.set_column('A:A', 10)  # ID column
                sheet_clusters.set_column('B:B', 30)  # Name column
                sheet_clusters.set_column('C:C', 20)  # Date column
                sheet_clusters.set_column('D:E', 15)  # Threshold and type
                sheet_clusters.set_column('F:G', 15)  # Count columns
                sheet_clusters.set_column('H:H', 40)  # Documents list
                sheet_clusters.set_column('I:I', 30)  # Paragraph types
                
        logger.info(f"Enhanced data export completed to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting to Excel: {str(e)}", exc_info=True)
        return False

def main():
    """Main function to handle command-line execution."""
    parser = argparse.ArgumentParser(description='Export Paragraph Analyzer data to Excel')
    parser.add_argument('--db-url', default="postgresql://paragraph_user:pass@localhost/paragraph_analyzer",
                       help='Database connection URL')
    parser.add_argument('--output', default=None,
                       help='Output Excel file path (default: paragraph_analysis_YYYYMMDD_HHMMSS.xlsx)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_level)
    
    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"paragraph_analysis_{timestamp}.xlsx"
    
    logger.info(f"Starting Excel export with database URL: {args.db_url}")
    logger.info(f"Output file: {args.output}")
    
    # Run export
    success = export_to_excel(args.db_url, args.output, logger)
    
    if success:
        logger.info("Export completed successfully")
        print(f"Export completed successfully to {args.output}")
        return 0
    else:
        logger.error("Export failed")
        print("Export failed. See log file for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())