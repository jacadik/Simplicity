import os
import pandas as pd
from typing import List, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import or_, func

from .models import Document, Paragraph, Tag, SimilarityResult, Cluster
from .base_manager import BaseManager

class ExportManager:
    """
    Manages data export operations for the database.
    """
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the export manager.
        
        Args:
            base_manager: Base manager instance for database operations
        """
        self.base_manager = base_manager
        self.logger = base_manager.logger
    
    def export_to_excel(self, output_path: str) -> bool:
        """
        Export database contents to Excel with updated similarity fields.
        
        Args:
            output_path: Path where the Excel file will be saved
            
        Returns:
            Boolean indicating success
        """
        self.logger.info(f"Exporting data to Excel: {output_path}")
        
        try:
            # Ensure the directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.logger.info(f"Created directory: {output_dir}")
            
            def db_export_to_excel(session: Session) -> tuple:
                # Get paragraphs with document info and tags
                paragraphs_data = []
                
                # Query paragraphs with document info
                results = session.query(
                    Paragraph, 
                    Document.filename, 
                    Document.upload_date
                ).join(
                    Document
                ).order_by(
                    Document.filename, 
                    Paragraph.position
                ).all()
                
                # Process paragraphs and include tags
                for para, filename, upload_date in results:
                    # Get tags as comma-separated string
                    tags_str = ', '.join([tag.name for tag in para.tags]) if para.tags else None
                    
                    paragraphs_data.append({
                        'id': para.id,
                        'content': para.content,
                        'paragraph_type': para.paragraph_type,
                        'header_content': para.header_content,
                        'filename': filename,
                        'upload_date': upload_date,
                        'tags': tags_str
                    })
                
                # Get similarity data with both similarity scores
                similarity_data = []
                
                # Query similarity results with paragraph content and document names
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
                ).filter(
                    SimilarityResult.content_similarity_score >= 0.8
                ).all()
                
                # Get paragraph2 and document2 data
                for sim, para1_content, doc1_filename in similarity_results:
                    para2 = session.query(Paragraph).get(sim.paragraph2_id)
                    if para2:
                        doc2 = session.query(Document).get(para2.document_id)
                        
                        similarity_data.append({
                            'content_similarity_score': sim.content_similarity_score,  # Renamed field
                            'text_similarity_score': sim.text_similarity_score,        # New field
                            'similarity_type': sim.similarity_type,
                            'paragraph1_content': para1_content,
                            'document1': doc1_filename,
                            'paragraph2_content': para2.content,
                            'document2': doc2.filename if doc2 else 'Unknown'
                        })
                
                return paragraphs_data, similarity_data
            
            paragraphs_data, similarity_data = self.base_manager._with_session(db_export_to_excel)
            
            # Convert to DataFrames
            paragraphs_df = pd.DataFrame(paragraphs_data)
            similarity_df = pd.DataFrame(similarity_data)
            
            # Write to Excel file
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                # Paragraphs sheet
                paragraphs_df.to_excel(writer, sheet_name='Paragraphs', index=False)
                
                # Similarities sheet - now with both similarity metrics
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
                    sheet2.set_column('A:A', 15)  # Content Similarity column
                    sheet2.set_column('B:B', 15)  # Text Similarity column
                    sheet2.set_column('C:C', 15)  # Type column
                    sheet2.set_column('D:E', 80)  # Content columns
                    sheet2.set_column('F:G', 30)  # Document columns
            
            self.logger.info(f"Data exported successfully to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting to Excel: {str(e)}", exc_info=True)
            return False
