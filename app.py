import os
import logging
from logging.handlers import RotatingFileHandler
import tempfile
import shutil
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, g
from werkzeug.utils import secure_filename

from document_parser import DocumentParser
from similarity_analyzer import SimilarityAnalyzer
from database_manager import DatabaseManager, Document, Paragraph, Tag, SimilarityResult

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paragraph_analyzer.log')

# PostgreSQL connection URL - update with your credentials
DB_URL = "postgresql://paragraph_user:pass@localhost/paragraph_analyzer"

# For SQLite fallback (if needed):
# DB_URL = f"sqlite:///{os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paragraph_analyzer.db')}"

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
app.secret_key = 'paragraph_analyzer_secret_key'  # For flash messages

# Add before_request function to detect AJAX requests
@app.before_request
def before_request():
    # Store XHR status in g object
    g.is_xhr = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

# Configure logging
def setup_logging(app):
    """Set up detailed logging configuration."""
    if not app.debug:
        # Create log handler for writing to file with rotation
        file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10485760, backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        # Also log to stderr
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        app.logger.addHandler(console_handler)
        
    app.logger.setLevel(logging.INFO)
    app.logger.info('Paragraph Analyzer startup')
    
    return app.logger

logger = setup_logging(app)

# Initialize components
document_parser = DocumentParser(logging_level='INFO')
similarity_analyzer = SimilarityAnalyzer(logging_level='INFO')
db_manager = DatabaseManager(DB_URL, logging_level='INFO')

# Utility functions
def allowed_file(filename):
    """Check if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with document management."""
    documents = db_manager.get_documents()
    return render_template('index.html', documents=documents)

@app.route('/upload', methods=['POST'])
def upload_documents():
    """Handle document upload."""
    if 'files[]' not in request.files:
        if g.is_xhr:  # Check if it's an AJAX request using g object
            return jsonify({'success': False, 'message': 'No files selected'})
        flash('No files selected', 'danger')
        return redirect(url_for('index'))
        
    files = request.files.getlist('files[]')
    
    if not files or files[0].filename == '':
        if g.is_xhr:
            return jsonify({'success': False, 'message': 'No files selected'})
        flash('No files selected', 'danger')
        return redirect(url_for('index'))
    
    success_count = 0
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Add timestamp to filename if it already exists
            if os.path.exists(file_path):
                name, ext = os.path.splitext(filename)
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                filename = f"{name}_{timestamp}{ext}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the uploaded file
            file.save(file_path)
            
            # Add document to database
            file_type = filename.rsplit('.', 1)[1].lower()
            doc_id = db_manager.add_document(filename, file_type, file_path)
            
            if doc_id > 0:
                # Parse document and extract paragraphs
                paragraphs = document_parser.parse_document(file_path, doc_id)
                
                if paragraphs:
                    # Add paragraphs to database
                    paragraph_ids = db_manager.add_paragraphs(paragraphs)
                    if paragraph_ids:
                        success_count += 1
                        logger.info(f"Processed {len(paragraphs)} paragraphs from {filename}")
                    else:
                        logger.warning(f"Failed to add paragraphs from {filename}")
                else:
                    logger.warning(f"No paragraphs extracted from {filename}")
            else:
                logger.error(f"Failed to add document to database: {filename}")
    
    if g.is_xhr:
        return jsonify({
            'success': success_count > 0,
            'message': f'Successfully uploaded and processed {success_count} document(s)' if success_count > 0 else 'Failed to process uploaded documents'
        })
    
    if success_count > 0:
        flash(f'Successfully uploaded and processed {success_count} document(s)', 'success')
    else:
        flash('Failed to process uploaded documents', 'danger')
    
    return redirect(url_for('index'))

@app.route('/upload-folder', methods=['POST'])
def upload_folder():
    """Handle folder upload via path."""
    folder_path = request.form.get('folder_path', '').strip()
    
    if not folder_path or not os.path.isdir(folder_path):
        flash('Invalid folder path', 'danger')
        return redirect(url_for('index'))
    
    # Get all allowed files in the folder
    files = []
    for filename in os.listdir(folder_path):
        if allowed_file(filename):
            files.append(os.path.join(folder_path, filename))
    
    if not files:
        flash('No valid files found in the folder', 'danger')
        return redirect(url_for('index'))
    
    success_count = 0
    for file_path in files:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Add timestamp to filename if it already exists
        if os.path.exists(dest_path):
            name, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            new_filename = f"{name}_{timestamp}{ext}"
            dest_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        else:
            new_filename = filename
        
        # Copy the file
        shutil.copy2(file_path, dest_path)
        
        # Add document to database
        file_type = filename.rsplit('.', 1)[1].lower()
        doc_id = db_manager.add_document(new_filename, file_type, dest_path)
        
        if doc_id > 0:
            # Parse document and extract paragraphs
            paragraphs = document_parser.parse_document(dest_path, doc_id)
            
            if paragraphs:
                # Add paragraphs to database
                paragraph_ids = db_manager.add_paragraphs(paragraphs)
                if paragraph_ids:
                    success_count += 1
                    logger.info(f"Processed {len(paragraphs)} paragraphs from {new_filename}")
                else:
                    logger.warning(f"Failed to add paragraphs from {new_filename}")
            else:
                logger.warning(f"No paragraphs extracted from {new_filename}")
        else:
            logger.error(f"Failed to add document to database: {new_filename}")
    
    if success_count > 0:
        flash(f'Successfully processed {success_count} document(s) from folder', 'success')
    else:
        flash('Failed to process documents from folder', 'danger')
    
    return redirect(url_for('index'))

@app.route('/delete/<int:document_id>')
def delete_document(document_id):
    """Delete a specific document."""
    if db_manager.delete_document(document_id):
        flash('Document deleted successfully', 'success')
    else:
        flash('Failed to delete document', 'danger')
    
    return redirect(url_for('index'))

@app.route('/clear')
def clear_database():
    """Clear all data."""
    if db_manager.clear_database():
        flash('All data cleared successfully', 'success')
    else:
        flash('Failed to clear data', 'danger')
    
    return redirect(url_for('index'))

@app.route('/paragraphs')
def view_paragraphs():
    """View paragraphs with filtering options."""
    document_id = request.args.get('document_id', type=int)
    show_all_duplicates = request.args.get('show_all_duplicates', type=int, default=0)
    
    # When viewing a specific document, or when explicitly requested, don't collapse duplicates
    collapse_duplicates = not (document_id is not None or show_all_duplicates == 1)
    
    paragraphs = db_manager.get_paragraphs(document_id, collapse_duplicates=collapse_duplicates)
    documents = db_manager.get_documents()
    tags = db_manager.get_tags()
    
    return render_template(
        'paragraphs.html', 
        paragraphs=paragraphs, 
        documents=documents, 
        tags=tags, 
        selected_document=document_id,
        show_all_duplicates=show_all_duplicates
    )

@app.route('/document/<int:document_id>')
def view_document(document_id):
    """View a document with its extracted paragraphs."""
    # Get document information
    session = db_manager.Session()
    document = session.query(Document).get(document_id)
    
    if not document:
        flash('Document not found', 'danger')
        return redirect(url_for('index'))
    
    # Get paragraphs for this document
    paragraphs = db_manager.get_paragraphs(document_id, collapse_duplicates=False)
    
    # Determine file type for proper rendering
    file_type = document.file_type.lower()
    
    return render_template(
        'document_view.html',
        document=document,
        paragraphs=paragraphs,
        file_type=file_type
    )

@app.route('/serve-document/<int:document_id>')
def serve_document(document_id):
    """Serve the document file for viewing."""
    session = db_manager.Session()
    document = session.query(Document).get(document_id)
    
    if not document or not os.path.exists(document.file_path):
        flash('Document not found', 'danger')
        return redirect(url_for('index'))
    
    return send_file(
        document.file_path,
        as_attachment=False,
        download_name=document.filename
    )
	
@app.route('/similarity')
def view_similarity():
    """View similarity analysis with threshold adjustment."""
    threshold = request.args.get('threshold', type=float, default=0.8)
    similarities = db_manager.get_similar_paragraphs(threshold)
    
    return render_template('similarity.html', similarities=similarities, threshold=threshold)

@app.route('/analyze-similarity', methods=['POST'])
def analyze_similarity():
    """Run similarity analysis on paragraphs."""
    threshold = float(request.form.get('threshold', 0.8))
    
    # Get all paragraphs
    paragraphs = db_manager.get_paragraphs()
    
    if not paragraphs:
        flash('No paragraphs available for analysis', 'warning')
        return redirect(url_for('view_similarity'))
    
    # Prepare data for similarity analysis
    para_data = []
    for para in paragraphs:
        para_data.append({
            'id': para['id'],
            'content': para['content'],
            'doc_id': para['document_id']
        })
    
    # Find exact matches first
    exact_matches = similarity_analyzer.find_exact_matches(para_data)
    
    # Find similar paragraphs
    similar_paragraphs = similarity_analyzer.find_similar_paragraphs(para_data, threshold)
    
    # Combine results
    all_results = exact_matches + similar_paragraphs
    
    if all_results:
        # Save results to database
        db_manager.add_similarity_results(all_results)
        flash(f'Found {len(exact_matches)} exact matches and {len(similar_paragraphs)} similar paragraphs', 'success')
    else:
        flash('No similarities found', 'info')
    
    return redirect(url_for('view_similarity', threshold=threshold))

@app.route('/tags')
def manage_tags():
    """Manage tags."""
    tags = db_manager.get_tags()
    return render_template('tags.html', tags=tags)

@app.route('/add-tag', methods=['POST'])
def add_tag():
    """Add a new tag."""
    name = request.form.get('name', '').strip()
    color = request.form.get('color', '#000000')
    
    if not name:
        flash('Tag name is required', 'danger')
        return redirect(url_for('manage_tags'))
    
    tag_id = db_manager.add_tag(name, color)
    
    if tag_id > 0:
        flash(f'Tag "{name}" added successfully', 'success')
    else:
        flash('Failed to add tag', 'danger')
    
    return redirect(url_for('manage_tags'))

@app.route('/delete-tag', methods=['POST'])
def delete_tag():
    """Delete a tag and remove all its associations."""
    tag_id = request.form.get('tag_id', type=int)
    
    if not tag_id:
        if g.is_xhr:
            return jsonify({'success': False, 'message': 'Invalid tag ID'})
        flash('Invalid tag ID', 'danger')
        return redirect(url_for('manage_tags'))
    
    success = db_manager.delete_tag(tag_id)
    
    # Check if it's an AJAX request
    if g.is_xhr:
        return jsonify({
            'success': success,
            'message': 'Tag deleted successfully' if success else 'Failed to delete tag'
        })
    
    if success:
        flash('Tag deleted successfully', 'success')
    else:
        flash('Failed to delete tag', 'danger')
    
    return redirect(url_for('manage_tags'))

@app.route('/tag-paragraph', methods=['POST'])
def tag_paragraph():
    """Tag a paragraph."""
    paragraph_id = request.form.get('paragraph_id', type=int)
    tag_id = request.form.get('tag_id', type=int)
    
    if not paragraph_id or not tag_id:
        return jsonify({'success': False, 'message': 'Invalid parameters'})
    
    if db_manager.tag_paragraph(paragraph_id, tag_id):
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'message': 'Failed to tag paragraph'})

@app.route('/untag-paragraph', methods=['POST'])
def untag_paragraph():
    """Remove a tag from a paragraph."""
    paragraph_id = request.form.get('paragraph_id', type=int)
    tag_id = request.form.get('tag_id', type=int)
    
    if not paragraph_id or not tag_id:
        return jsonify({'success': False, 'message': 'Invalid parameters'})
    
    if db_manager.untag_paragraph(paragraph_id, tag_id):
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'message': 'Failed to untag paragraph'})

@app.route('/get-tags')
def get_tags_json():
    """Get all tags as JSON."""
    tags = db_manager.get_tags()
    return jsonify(tags)

# Add these routes to app.py

@app.route('/clusters')
def view_clusters():
    """View all paragraph clusters."""
    clusters = db_manager.get_clusters()
    return render_template('clusters.html', clusters=clusters)

@app.route('/clusters/<int:cluster_id>')
def view_cluster(cluster_id):
    """View paragraphs in a specific cluster."""
    cluster = next((c for c in db_manager.get_clusters() if c['id'] == cluster_id), None)
    if not cluster:
        flash('Cluster not found', 'danger')
        return redirect(url_for('view_clusters'))
        
    paragraphs = db_manager.get_cluster_paragraphs(cluster_id)
    return render_template('cluster_paragraphs.html', cluster=cluster, paragraphs=paragraphs)

@app.route('/create-clusters', methods=['POST'])
def create_clusters():
    """Create paragraph clusters based on similarity analysis."""
    threshold = float(request.form.get('threshold', 0.8))
    
    # Get all similarity results
    similarities = db_manager.get_similar_paragraphs(threshold)
    
    if not similarities:
        flash('No similarities found for clustering', 'warning')
        return redirect(url_for('view_similarity'))
    
    # Convert database results to SimilarityResult objects for the algorithm
    similarity_results = []
    for sim in similarities:
        result = SimilarityResult(
            paragraph1_id=sim['paragraph1_id'],
            paragraph2_id=sim['paragraph2_id'],
            paragraph1_content=sim['para1_content'],
            paragraph2_content=sim['para2_content'],
            paragraph1_doc_id=sim['para1_doc_id'],
            paragraph2_doc_id=sim['para2_doc_id'],
            similarity_score=sim['similarity_score'],
            similarity_type=sim['similarity_type']
        )
        similarity_results.append(result)
    
    # Cluster the paragraphs
    clusters = similarity_analyzer.cluster_paragraphs(similarity_results, threshold)
    
    if not clusters:
        flash('No clusters found', 'warning')
        return redirect(url_for('view_similarity'))
    
    # Save clusters to database
    cluster_count = 0
    for cluster in clusters:
        cluster_id = db_manager.create_cluster(
            name=cluster['name'],
            description=cluster['description'],
            similarity_threshold=cluster['similarity_threshold']
        )
        
        if cluster_id > 0:
            if db_manager.add_paragraphs_to_cluster(cluster_id, cluster['paragraph_ids']):
                cluster_count += 1
    
    if cluster_count > 0:
        flash(f'Successfully created {cluster_count} clusters', 'success')
    else:
        flash('Failed to create clusters', 'danger')
    
    return redirect(url_for('view_clusters'))

@app.route('/delete-cluster/<int:cluster_id>')
def delete_cluster(cluster_id):
    """Delete a specific cluster."""
    if db_manager.delete_cluster(cluster_id):
        flash('Cluster deleted successfully', 'success')
    else:
        flash('Failed to delete cluster', 'danger')
    
    return redirect(url_for('view_clusters'))
	
@app.route('/export')
def export_data():
    """Export data to Excel."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp:
            temp_path = temp.name
        
        # Export data to the temporary file
        if db_manager.export_to_excel(temp_path):
            return send_file(
                temp_path,
                as_attachment=True,
                download_name=f'paragraph_analysis_{datetime.now().strftime("%Y%m%d%H%M%S")}.xlsx',
                max_age=0
            )
        else:
            flash('Failed to export data', 'danger')
            return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}", exc_info=True)
        flash('An error occurred during export', 'danger')
        return redirect(url_for('index'))

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    flash('File too large', 'danger')
    return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(error):
    """Handle 404 error."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(error):
    """Handle 500 error."""
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)