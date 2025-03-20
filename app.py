import os
import logging
from logging.handlers import RotatingFileHandler
import tempfile
import shutil
import time
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, g
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit

from utils.document_parser import DocumentParser
from utils.similarity_analyzer import SimilarityAnalyzer, SimilarityResult as AnalyzerSimilarityResult
from utils.database_manager import DatabaseManager, Document, Paragraph, Tag, SimilarityResult, Cluster, cluster_paragraphs
from utils.document_metadata_extractor import DocumentMetadataExtractor
from utils.excel_exporter import export_to_excel
from utils.thread_pool_manager import ThreadPoolManager
from utils.document_batch_processor import DocumentBatchProcessor

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

# Initialize Flask-SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

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
    
    # Add console handler if not already added
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger
    
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
metadata_extractor = DocumentMetadataExtractor(logging_level='INFO')

# Initialize batch processor
batch_processor = DocumentBatchProcessor(
    db_manager=db_manager,
    document_parser=document_parser,
    metadata_extractor=metadata_extractor,
    upload_folder=app.config['UPLOAD_FOLDER'],
    max_workers=None,  # Use default (CPU count + 4)
    logging_level='INFO'
)

# Store batch processing results for progress tracking
batch_results = []
batch_start_time = 0

# Socket.IO progress event handler
def progress_callback(completed, total, result):
    """Send progress updates via WebSocket."""
    socketio.emit('upload_progress', {
        'completed': completed,
        'total': total,
        'current_file': result.get('filename', 'Unknown'),
        'success': result.get('success', False),
        'progress_percent': int((completed / total) * 100),
        'stats': {
            'success_count': sum(1 for r in batch_results[:completed] if r.get('success', False)),
            'total': total,
            'avg_time': (time.time() - batch_start_time) / max(completed, 1)
        }
    })

# Utility functions
def allowed_file(filename):
    """Check if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with document management and pagination."""
    # Get pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Number of documents per page
    
    # Get all documents first
    documents = db_manager.get_documents()
    
    # Calculate total pages
    total_documents = len(documents)
    total_pages = (total_documents + per_page - 1) // per_page  # Ceiling division
    
    # Ensure page is within valid range
    if page < 1:
        page = 1
    elif page > total_pages and total_pages > 0:
        page = total_pages
    
    # Get slice of documents for current page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_documents = documents[start_idx:end_idx] if documents else []
    
    # Get additional statistics
    # Here you could add queries to get total paragraphs, duplicates, tags etc.
    total_paragraphs = 0  # Replace with actual query
    duplicates = 0        # Replace with actual query
    tags = 0              # Replace with actual query
    
    return render_template(
        'index.html',
        documents=documents,            # All documents (for stats)
        paginated_documents=paginated_documents,  # Documents for current page
        page=page,
        total_pages=total_pages,
        per_page=per_page,
        total_paragraphs=total_paragraphs,
        duplicates=duplicates,
        tags=tags
    )

@app.route('/upload', methods=['POST'])
def upload_documents():
    """Handle document upload with multi-threaded processing."""
    global batch_results, batch_start_time
    
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
    
    # Reset batch results and start time
    batch_results = []
    batch_start_time = time.time()
    
    # Define the progress callback function
    def progress_callback(completed, total, result):
        """Send progress updates via Socket.IO."""
        app.logger.info(f"Progress: {completed}/{total} - Processing: {result.get('filename', 'Unknown')}")
        
        # Calculate success count
        success_count = sum(1 for r in batch_results[:completed] if r.get('success', False))
        
        # Calculate average time per file
        elapsed = time.time() - batch_start_time
        avg_time = elapsed / max(completed, 1)
        
        # Emit progress update
        socketio.emit('upload_progress', {
            'completed': completed,
            'total': total,
            'current_file': result.get('filename', 'Unknown'),
            'success': result.get('success', False),
            'progress_percent': int((completed / total) * 100),
            'stats': {
                'success_count': success_count,
                'total': total,
                'avg_time': avg_time
            }
        })
    
    # Process files using multi-threaded batch processor
    results = batch_processor.process_uploaded_files(files, progress_callback=progress_callback)
    
    # Store results for progress tracking
    batch_results = results.get('results', [])
    
    # Send completion event
    socketio.emit('upload_complete', {
        'processed': results['processed'],
        'total': results['total'],
        'elapsed_time': results['elapsed_time']
    })
    
    if g.is_xhr:
        return jsonify({
            'success': results['success'],
            'message': f'Successfully uploaded and processed {results["processed"]} document(s)' if results['success'] else 'Failed to process uploaded documents',
            'total': results['total'],
            'processed': results['processed'],
            'elapsed_time': results['elapsed_time']
        })
    
    if results['success']:
        flash(f'Successfully uploaded and processed {results["processed"]} document(s)', 'success')
    else:
        flash('Failed to process uploaded documents', 'danger')
    
    return redirect(url_for('index'))

# Add this new route to scan a folder before processing
@app.route('/upload-folder', methods=['POST'])
def upload_folder():
    """Handle folder upload via path with multi-threaded processing."""
    global batch_results, batch_start_time
    
    folder_path = request.form.get('folder_path', '').strip()
    
    if not folder_path or not os.path.isdir(folder_path):
        if g.is_xhr:
            return jsonify({'success': False, 'message': 'Invalid folder path'})
        flash('Invalid folder path', 'danger')
        return redirect(url_for('index'))
    
    # First scan the folder to count eligible files
    eligible_files = []
    total_scanned = 0
    
    app.logger.info(f"Scanning folder: {folder_path}")
    
    try:
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                total_scanned += 1
                if allowed_file(filename):
                    source_path = os.path.join(root, filename)
                    eligible_files.append({
                        'filename': filename, 
                        'source_path': source_path
                    })
                
                # Send scanning progress update every 10 files
                if total_scanned % 10 == 0:
                    app.logger.info(f"Scanning progress: found {len(eligible_files)} eligible files out of {total_scanned} scanned")
                    socketio.emit('folder_scan_progress', {
                        'file_count': len(eligible_files),
                        'total_scanned': total_scanned,
                        'scanning': True
                    })
    except Exception as e:
        app.logger.error(f"Error scanning folder: {str(e)}")
        if g.is_xhr:
            return jsonify({'success': False, 'message': f'Error scanning folder: {str(e)}'})
        flash(f'Error scanning folder: {str(e)}', 'danger')
        return redirect(url_for('index'))
    
    # Final scan update
    socketio.emit('folder_scan_progress', {
        'file_count': len(eligible_files),
        'total_scanned': total_scanned,
        'scanning': False
    })
    
    if not eligible_files:
        if g.is_xhr:
            return jsonify({'success': False, 'message': 'No eligible files found in folder'})
        flash('No eligible files found in folder', 'danger')
        return redirect(url_for('index'))
    
    app.logger.info(f"Found {len(eligible_files)} eligible files in folder {folder_path}")
    
    # Reset batch results and start time
    batch_results = []
    batch_start_time = time.time()
    
    # Prepare documents for processing
    document_infos = []
    
    for file_info in eligible_files:
        filename = file_info['filename']
        source_path = file_info['source_path']
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
        try:
            shutil.copy2(source_path, dest_path)
            
            # Prepare document info for processing
            document_infos.append({
                'filename': new_filename,
                'file_path': dest_path,
                'file_type': filename.rsplit('.', 1)[1].lower()
            })
        except Exception as e:
            app.logger.error(f"Error copying file {source_path}: {str(e)}")
            # Continue with other files
    
    # Define a folder-specific progress callback
    def folder_progress_callback(completed, total, result):
        """Send progress updates via Socket.IO specifically for folder uploads."""
        app.logger.info(f"Folder progress: {completed}/{total} - Processing: {result.get('filename', 'Unknown')}")
        
        # Calculate success count
        success_count = sum(1 for r in batch_results[:completed] if r.get('success', False))
        
        # Calculate average time per file
        elapsed = time.time() - batch_start_time
        avg_time = elapsed / max(completed, 1)
        
        # Emit progress update
        socketio.emit('folder_upload_progress', {
            'completed': completed,
            'total': total,
            'current_file': result.get('filename', 'Unknown'),
            'success': result.get('success', False),
            'progress_percent': int((completed / total) * 100),
            'stats': {
                'success_count': success_count,
                'total': total,
                'avg_time': avg_time
            }
        })
    
    # Process documents using multi-threaded batch processor
    results = batch_processor.process_batch_documents(document_infos, progress_callback=folder_progress_callback)
    
    # Store results for progress tracking
    batch_results = results.get('results', [])
    
    # Send completion event
    socketio.emit('folder_upload_complete', {
        'processed': results['processed'],
        'total': results['total'],
        'elapsed_time': results['elapsed_time']
    })
    
    if g.is_xhr:
        return jsonify({
            'success': results['success'],
            'message': f'Successfully processed {results["processed"]} document(s) from folder' if results['success'] else 'Failed to process documents from folder',
            'total': results['total'],
            'processed': results['processed'],
            'elapsed_time': results['elapsed_time']
        })
    
    if results['success']:
        flash(f'Successfully processed {results["processed"]} document(s) from folder', 'success')
    else:
        flash('Failed to process documents from folder', 'danger')
    
    return redirect(url_for('index'))
    
    # Process documents using multi-threaded batch processor
    results = batch_processor.process_batch_documents(document_infos, progress_callback=folder_progress_callback)
    
    # Store results for progress tracking
    batch_results = results.get('results', [])
    
    # Send completion event
    socketio.emit('folder_upload_complete', {
        'processed': results['processed'],
        'total': results['total'],
        'elapsed_time': results['elapsed_time']
    })
    
    if g.is_xhr:
        return jsonify({
            'success': results['success'],
            'message': f'Successfully processed {results["processed"]} document(s) from folder' if results['success'] else 'Failed to process documents from folder',
            'total': results['total'],
            'processed': results['processed'],
            'elapsed_time': results['elapsed_time']
        })
    
    if results['success']:
        flash(f'Successfully processed {results["processed"]} document(s) from folder', 'success')
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
    
    # Create a dictionary to map filenames to document IDs for easy lookup
    filename_to_doc_id = {doc['filename']: doc['id'] for doc in documents}
    
    tags = db_manager.get_tags()
    
    return render_template(
        'paragraphs.html', 
        paragraphs=paragraphs, 
        documents=documents, 
        tags=tags, 
        selected_document=document_id,
        show_all_duplicates=show_all_duplicates,
        filename_to_doc_id=filename_to_doc_id  # Add this dictionary to template context
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
    
    # Get file metadata
    file_metadata = db_manager.get_document_file_metadata(document_id)
    
    # Determine file type for proper rendering
    file_type = document.file_type.lower()
    
    return render_template(
        'document_view.html',
        document=document,
        paragraphs=paragraphs,
        file_type=file_type,
        file_metadata=file_metadata
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
    # Get threshold as percentage (0-100), default to 80%
    threshold_pct = request.args.get('threshold', type=float, default=80.0)
    
    # Convert percentage to decimal (0-1) for database query
    threshold = threshold_pct / 100.0
    
    # Get similarities using the decimal threshold
    similarities = db_manager.get_similar_paragraphs(threshold)
    
    # Pass the percentage threshold to the template
    return render_template('similarity.html', similarities=similarities, threshold=threshold_pct)

@app.route('/analyze-similarity', methods=['POST'])
def analyze_similarity():
    """Run similarity analysis on paragraphs."""
    # Get threshold as percentage (0-100)
    threshold_pct = float(request.form.get('threshold', 80.0))
    
    # Convert percentage to decimal (0-1) for similarity analyzer
    threshold = threshold_pct / 100.0
    
    # Get all paragraphs - use collapse_duplicates=False to get ALL paragraphs
    paragraphs = db_manager.get_paragraphs(collapse_duplicates=False)
    
    if not paragraphs:
        flash('No paragraphs available for analysis', 'warning')
        return redirect(url_for('view_similarity'))
    
    app.logger.info(f"Retrieved {len(paragraphs)} paragraphs for similarity analysis")
    
    # Log document distribution for debugging
    doc_counts = {}
    for para in paragraphs:
        doc_id = para['document_id']
        if doc_id not in doc_counts:
            doc_counts[doc_id] = 0
        doc_counts[doc_id] += 1
    
    app.logger.info(f"Paragraph distribution by document: {doc_counts}")
    
    # Prepare data for similarity analysis
    para_data = []
    for para in paragraphs:
        para_data.append({
            'id': para['id'],
            'content': para['content'],
            'doc_id': para['document_id']
        })
    
    # Clear existing similarity results before finding new ones
    db_manager.clear_similarity_results()
    
    # Find exact matches first
    app.logger.info("Finding exact matches...")
    exact_matches = similarity_analyzer.find_exact_matches(para_data)
    app.logger.info(f"Found {len(exact_matches)} exact matches")
    
    # Find similar paragraphs with the converted threshold
    app.logger.info("Finding similar paragraphs...")
    similar_paragraphs = similarity_analyzer.find_similar_paragraphs(para_data, threshold)
    app.logger.info(f"Found {len(similar_paragraphs)} similar paragraphs")
    
    # Combine results
    all_results = exact_matches + similar_paragraphs
    
    if all_results:
        # Save results to database
        app.logger.info(f"Saving {len(all_results)} similarity results to database")
        db_manager.add_similarity_results(all_results)
        flash(f'Found {len(exact_matches)} exact matches and {len(similar_paragraphs)} similar paragraphs', 'success')
    else:
        flash('No similarities found', 'info')
    
    # Redirect to similarity view with the same threshold percentage
    return redirect(url_for('view_similarity', threshold=threshold_pct))

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
    try:
        paragraph_id = request.form.get('paragraph_id')
        tag_id = request.form.get('tag_id')
        
        # Explicitly check for the tag_all_duplicates parameter
        tag_all_duplicates = request.form.get('tag_all_duplicates', 'false').lower() == 'true'
        
        if not paragraph_id or not tag_id:
            return jsonify({'success': False, 'message': 'Missing parameters'})
        
        # Convert to integers
        paragraph_id = int(paragraph_id)
        tag_id = int(tag_id)
        
        # Debug logging
        app.logger.info(f"Tagging paragraph {paragraph_id} with tag {tag_id}, tag_all_duplicates: {tag_all_duplicates}")
        
        # Call database manager with the tag_all_duplicates parameter
        success = db_manager.tag_paragraph(paragraph_id, tag_id, tag_all_duplicates)
        
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Failed to add tag'})
    except Exception as e:
        app.logger.error(f"Error in tag_paragraph: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)})

# Route handler for untagging paragraphs
@app.route('/untag-paragraph', methods=['POST'])
def untag_paragraph():
    try:
        paragraph_id = request.form.get('paragraph_id')
        tag_id = request.form.get('tag_id')
        
        # Explicitly check for the untag_all_duplicates parameter
        untag_all_duplicates = request.form.get('untag_all_duplicates', 'false').lower() == 'true'
        
        if not paragraph_id or not tag_id:
            return jsonify({'success': False, 'message': 'Missing parameters'})
        
        # Convert to integers
        paragraph_id = int(paragraph_id)
        tag_id = int(tag_id)
        
        # Debug logging
        app.logger.info(f"Untagging paragraph {paragraph_id} with tag {tag_id}, untag_all_duplicates: {untag_all_duplicates}")
        
        # Call database manager with the untag_all_duplicates parameter
        success = db_manager.untag_paragraph(paragraph_id, tag_id, untag_all_duplicates)
        
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Failed to remove tag'})
    except Exception as e:
        app.logger.error(f"Error in untag_paragraph: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)})


@app.route('/get-tags')
def get_tags_json():
    """Get all tags as JSON."""
    tags = db_manager.get_tags()
    return jsonify(tags)

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
    similarity_type = request.form.get('similarity_type', 'content')  # Default to content similarity
    
    # Clear all existing clusters before creating new ones
    if not db_manager.clear_all_clusters():
        flash('Failed to clear existing clusters', 'warning')
    
    # Get all similarity results
    similarities = db_manager.get_similar_paragraphs(threshold)
    
    if not similarities:
        flash('No similarities found for clustering', 'warning')
        return redirect(url_for('view_similarity'))
    
    # Convert database results to AnalyzerSimilarityResult objects for the algorithm
    similarity_results = []
    for sim in similarities:
        result = AnalyzerSimilarityResult(
            paragraph1_id=sim['paragraph1_id'],
            paragraph2_id=sim['paragraph2_id'],
            paragraph1_content=sim['para1_content'],
            paragraph2_content=sim['para2_content'],
            paragraph1_doc_id=sim['para1_doc_id'],
            paragraph2_doc_id=sim['para2_doc_id'],
            content_similarity_score=sim['content_similarity_score'],
            text_similarity_score=sim['text_similarity_score'],
            similarity_type=sim['similarity_type']
        )
        similarity_results.append(result)
    
    # Cluster the paragraphs using the specified similarity type
    clusters = similarity_analyzer.cluster_paragraphs(similarity_results, threshold, similarity_type)
    
    if not clusters:
        flash('No clusters found', 'warning')
        return redirect(url_for('view_similarity'))
    
    # Save clusters to database
    cluster_count = 0
    for cluster in clusters:
        cluster_id = db_manager.create_cluster(
            name=cluster['name'],
            description=cluster['description'],
            similarity_threshold=cluster['similarity_threshold'],
            similarity_type=cluster['similarity_type']  # Store which similarity type was used
        )
        
        if cluster_id > 0:
            if db_manager.add_paragraphs_to_cluster(cluster_id, cluster['paragraph_ids']):
                cluster_count += 1
    
    if cluster_count > 0:
        flash(f'Successfully created {cluster_count} clusters using {similarity_type} similarity', 'success')
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
        
        # Generate a meaningful filename for the download
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        download_filename = f'paragraph_analysis_{timestamp}.xlsx'
        
        # Use the standalone excel_exporter if available
        try:
            from utils.excel_exporter import export_to_excel as standalone_export
            
            # Call the standalone exporter with our database URL
            if standalone_export(DB_URL, temp_path, logger):
                return send_file(
                    temp_path,
                    as_attachment=True,
                    download_name=download_filename,
                    max_age=0
                )
            else:
                flash('Failed to export data', 'danger')
                return redirect(url_for('index'))
                
        except ImportError:
            # Fall back to the database manager's export method
            logger.info("Standalone exporter not available, using database manager's export method")
            if db_manager.export_to_excel(temp_path):
                return send_file(
                    temp_path,
                    as_attachment=True,
                    download_name=download_filename,
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
    socketio.run(app, debug=True)
