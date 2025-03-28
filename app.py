import os
import logging
from logging.handlers import RotatingFileHandler
import tempfile
import shutil
import time
import threading
from queue import Queue
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, g
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit

from utils.document_parser import DocumentParser
from utils.similarity_analyzer import SimilarityAnalyzer, SimilarityResult as AnalyzerSimilarityResult
from utils.document_metadata_extractor import DocumentMetadataExtractor
from utils.excel_exporter import export_to_excel
from utils.thread_pool_manager import ThreadPoolManager
from utils.document_batch_processor import DocumentBatchProcessor
from utils.database.models import Document, Paragraph, Tag, SimilarityResult, Cluster, cluster_paragraphs
from utils.database.manager import DatabaseManager

# Initialize insert page extractor
from utils.insert_page_extractor import InsertPageExtractor
insert_page_extractor = InsertPageExtractor(logging_level='INFO')

# Import insert matcher (will be instantiated when needed)
from utils.insert_matcher import InsertMatcher, InsertMatchResult


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
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Create a thread-safe queue for progress updates
progress_queue = Queue()

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

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Socket.IO client connected from thread {threading.get_ident()}")
    
    # Start a background task to process the progress queue
    socketio.start_background_task(target=process_progress_queue)

def process_progress_queue():
    """Process the progress queue in a separate thread."""
    logger.info(f"Starting progress queue processor in thread {threading.get_ident()}")
    
    while True:
        if not progress_queue.empty():
            try:
                # Get progress data from queue
                data = progress_queue.get()
                event_type = data.pop('event_type', 'upload_progress')
                
                # Emit the event
                logger.debug(f"Emitting {event_type} event: {data}")
                socketio.emit(event_type, data)
            except Exception as e:
                logger.error(f"Error processing progress queue: {str(e)}", exc_info=True)
        
        # Sleep briefly to avoid high CPU usage
        socketio.sleep(0.1)

@socketio.on('echo')
def handle_echo(data):
    """Echo test for Socket.IO connectivity."""
    logger.info(f"Received echo request: {data}")
    socketio.emit('echo_response', {'received': data, 'time': time.time()})

# Thread-safe progress callback

def progress_callback(completed, total, result):
    """Send progress updates via Socket.IO."""
    app.logger.info(f"Progress: {completed}/{total} - Processing: {result.get('filename', 'Unknown')}")
    
    # Add the missing update to batch_results
    if len(batch_results) < completed:
        batch_results.append(result)
    elif len(batch_results) >= completed:
        batch_results[completed-1] = result
    
    # Calculate success count
    success_count = sum(1 for r in batch_results[:completed] if r.get('success', False))
    
    # Calculate average time per file
    elapsed = time.time() - batch_start_time
    avg_time = elapsed / max(completed, 1)
    
    # Emit progress update with namespace
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

# Thread-safe completion callback
def completion_callback(data):
    """Queue a completion event."""
    logger.info(f"Completion callback from thread {threading.get_ident()}")
    
    try:
        # Queue completion event
        progress_queue.put({
            'event_type': 'upload_complete',
            'processed': data['processed'],
            'total': data['total'],
            'elapsed_time': data['elapsed_time']
        })
    except Exception as e:
        logger.error(f"Error in completion callback: {str(e)}", exc_info=True)

# Utility functions
def allowed_file(filename):
    """Check if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with application overview and process explanation."""
    return render_template('index.html')

@app.route('/documents')
def documents():
    """Documents page with document management and pagination."""
    # Get pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Number of documents per page
    
    # Get paginated documents directly from the database
    paginated_documents, total_documents = db_manager.get_paginated_documents(page, per_page)
    
    # Calculate total pages
    total_pages = (total_documents + per_page - 1) // per_page  # Ceiling division
    
    # Ensure page is within valid range
    if page < 1:
        page = 1
    elif page > total_pages and total_pages > 0:
        page = total_pages
    
    return render_template(
        'documents.html',
        documents=paginated_documents,  # Only pass the paginated documents
        paginated_documents=paginated_documents,
        page=page,
        total_pages=total_pages,
        per_page=per_page,
        total_documents=total_documents,
        # Remove statistics from here - they'll be loaded asynchronously
        # Add icons for the tiles
        icons={
            'documents': 'bi-file-earmark-text',
            'paragraphs': 'bi-paragraph',
            'duplicates': 'bi-files',
            'unique': 'bi-fingerprint'
        }
    )

@app.route('/upload', methods=['POST'])
def upload_documents():
    """Handle document upload with multi-threaded processing."""
    global batch_results, batch_start_time
    
    # Log current thread for debugging
    logger.info(f"Upload handler running in thread {threading.get_ident()}")
    
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
    
    # Log the start of processing
    logger.info(f"Starting to process {len(files)} files")
    
    # Process files using multi-threaded batch processor
    results = batch_processor.process_uploaded_files(files, progress_callback=progress_callback)
    
    # Store results for progress tracking
    batch_results = results.get('results', [])
    
    # Queue completion event instead of direct emission
    completion_callback(results)
    
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

# Testing route for Socket.IO
@app.route('/test-socket')
def test_socket():
    """Test route for Socket.IO connectivity."""
    logger.info(f"Socket.IO test from thread {threading.get_ident()}")
    
    # Queue a test message
    progress_queue.put({
        'event_type': 'test_event',
        'message': 'Test event from server',
        'time': time.time()
    })
    
    return "Socket.IO test event sent. Check browser console to see if it was received."

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
    
    logger.info(f"Scanning folder: {folder_path}")
    
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
                    logger.info(f"Scanning progress: found {len(eligible_files)} eligible files out of {total_scanned} scanned")
                    # Queue folder scan progress update
                    progress_queue.put({
                        'event_type': 'folder_scan_progress',
                        'file_count': len(eligible_files),
                        'total_scanned': total_scanned,
                        'scanning': True
                    })
    except Exception as e:
        logger.error(f"Error scanning folder: {str(e)}")
        if g.is_xhr:
            return jsonify({'success': False, 'message': f'Error scanning folder: {str(e)}'})
        flash(f'Error scanning folder: {str(e)}', 'danger')
        return redirect(url_for('index'))
    
    # Final scan update
    progress_queue.put({
        'event_type': 'folder_scan_progress',
        'file_count': len(eligible_files),
        'total_scanned': total_scanned,
        'scanning': False
    })
    
    if not eligible_files:
        if g.is_xhr:
            return jsonify({'success': False, 'message': 'No eligible files found in folder'})
        flash('No eligible files found in folder', 'danger')
        return redirect(url_for('index'))
    
    logger.info(f"Found {len(eligible_files)} eligible files in folder {folder_path}")
    
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
            logger.error(f"Error copying file {source_path}: {str(e)}")
            # Continue with other files
    
    # Process documents using multi-threaded batch processor
    results = batch_processor.process_batch_documents(document_infos, progress_callback=progress_callback)
    
    # Store results for progress tracking
    batch_results = results.get('results', [])
    
    # Queue completion event
    progress_queue.put({
        'event_type': 'folder_upload_complete',
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
    
    logger.info(f"Retrieved {len(paragraphs)} paragraphs for similarity analysis")
    
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
    logger.info("Finding exact matches...")
    exact_matches = similarity_analyzer.find_exact_matches(para_data)
    logger.info(f"Found {len(exact_matches)} exact matches")
    
    # Find similar paragraphs with the converted threshold
    logger.info("Finding similar paragraphs...")
    similar_paragraphs = similarity_analyzer.find_similar_paragraphs(para_data, threshold)
    logger.info(f"Found {len(similar_paragraphs)} similar paragraphs")
    
    # Combine results
    all_results = exact_matches + similar_paragraphs
    
    if all_results:
        # Save results to database
        logger.info(f"Saving {len(all_results)} similarity results to database")
        db_manager.add_similarity_results(all_results)
        flash(f'Found {len(exact_matches)} exact matches and {len(similar_paragraphs)} similar paragraphs', 'success')
    else:
        flash('No similarities found', 'info')
    
    # Redirect to similarity view with the same threshold percentage
    return redirect(url_for('view_similarity', threshold=threshold_pct))

@app.route('/tags')
def manage_tags():
    """Manage tags with enhanced statistics."""
    # Get tags with usage counts
    tags = db_manager.get_tags()
    
    # Calculate additional statistics
    tagged_paragraphs = 0
    most_used_tag = None
    most_used_tag_count = 0
    
    if tags:
        # Find total tagged paragraphs (this might include duplicates if paragraphs have multiple tags)
        tagged_paragraphs = sum(tag['usage_count'] for tag in tags)
        
        # Find the most used tag (both name and count)
        most_used_tag = max(tags, key=lambda tag: tag['usage_count']) if tags else None
        most_used_tag_count = most_used_tag['usage_count'] if most_used_tag else 0
    
    return render_template(
        'tags.html', 
        tags=tags,
        tagged_paragraphs=tagged_paragraphs,
        most_used_tag=most_used_tag,
        most_used_tag_count=most_used_tag_count
    )

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
        logger.info(f"Tagging paragraph {paragraph_id} with tag {tag_id}, tag_all_duplicates: {tag_all_duplicates}")
        
        # Call database manager with the tag_all_duplicates parameter
        success = db_manager.tag_paragraph(paragraph_id, tag_id, tag_all_duplicates)
        
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Failed to add tag'})
    except Exception as e:
        logger.error(f"Error in tag_paragraph: {str(e)}", exc_info=True)
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
        logger.info(f"Untagging paragraph {paragraph_id} with tag {tag_id}, untag_all_duplicates: {untag_all_duplicates}")
        
        # Call database manager with the untag_all_duplicates parameter
        success = db_manager.untag_paragraph(paragraph_id, tag_id, untag_all_duplicates)
        
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Failed to remove tag'})
    except Exception as e:
        logger.error(f"Error in untag_paragraph: {str(e)}", exc_info=True)
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


@app.route('/inserts', methods=['GET'])
def view_inserts():
    """View and manage inserts."""
    inserts = db_manager.get_inserts()
    return render_template('inserts.html', inserts=inserts)

@app.route('/upload-insert', methods=['POST'])
def upload_insert():
    """Handle insert upload with custom name."""
    if 'insert_file' not in request.files:
        flash('No file selected', 'danger')
        return redirect(url_for('view_inserts'))
        
    file = request.files['insert_file']
    insert_name = request.form.get('insert_name', '').strip()
    
    if not file or file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('view_inserts'))
        
    if not insert_name:
        flash('Insert name is required', 'danger')
        return redirect(url_for('view_inserts'))
    
    if not allowed_file(file.filename):
        flash('Invalid file type. Allowed types: PDF, DOC, DOCX', 'danger')
        return redirect(url_for('view_inserts'))
    
    try:
        # Process the insert
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Add timestamp to filename if it already exists
        if os.path.exists(file_path):
            name, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"{name}_{timestamp}{ext}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        file.save(file_path)
        
        # Extract pages from the insert
        file_type = filename.rsplit('.', 1)[1].lower()
        
        # Add insert to database
        insert_id = db_manager.add_insert(insert_name, filename, file_type, file_path)
        
        if insert_id > 0:
            # Extract pages and store them
            pages = insert_page_extractor.extract_pages(file_path, insert_id)
            
            if pages:
                db_manager.add_insert_pages(pages)
                flash(f'Insert "{insert_name}" added successfully with {len(pages)} pages', 'success')
            else:
                flash(f'Insert "{insert_name}" added but no pages could be extracted', 'warning')
        else:
            flash('Failed to add insert', 'danger')
            
    except Exception as e:
        logger.error(f"Error processing insert: {str(e)}", exc_info=True)
        flash(f'Error processing insert: {str(e)}', 'danger')
    
    return redirect(url_for('view_inserts'))

@app.route('/find-insert-matches/<int:insert_id>', methods=['GET'])
def find_insert_matches(insert_id):
    """Find documents that contain the given insert."""
    # Get the insert
    inserts = db_manager.get_inserts()
    insert = next((i for i in inserts if i['id'] == insert_id), None)
    
    if not insert:
        flash('Insert not found', 'danger')
        return redirect(url_for('view_inserts'))
    
    # Get the insert pages
    insert_pages = db_manager.get_insert_pages(insert_id)
    
    if not insert_pages:
        flash('No pages found for this insert', 'danger')
        return redirect(url_for('view_inserts'))
    
    # Get all documents
    documents = db_manager.get_documents()
    
    if not documents:
        flash('No documents available for comparison', 'warning')
        return render_template('insert_matches.html', insert=insert, matches=[])
    
    # For each document, get its pages
    document_pages = {}
    for doc in documents:
        # Get the document's paragraphs and organize them by page
        # This is a simplified approach - you may need to adapt this to your document structure
        paragraphs = db_manager.get_paragraphs(doc['id'], collapse_duplicates=False)
        
        # Group paragraphs by page
        pages = {}
        for para in paragraphs:
            page_num = para.get('page_number', 0)
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(para)
        
        # Convert to page content by joining paragraphs
        doc_pages = []
        for page_num, page_paras in sorted(pages.items()):
            content = '\n'.join(p['content'] for p in page_paras)
            doc_pages.append({
                'content': content,
                'page_number': page_num,
                'document_id': doc['id']
            })
        
        document_pages[doc['id']] = doc_pages
    
    # Create an insert matcher and find matches
    insert_matcher = InsertMatcher(
        similarity_analyzer=similarity_analyzer,
        similarity_threshold=0.3  # Adjust threshold as needed
    )
    
    matches = insert_matcher.find_insert_matches(
        insert_id=insert_id,
        insert_pages=insert_pages,
        documents=documents,
        document_pages=document_pages
    )
    
    return render_template('insert_matches.html', insert=insert, matches=matches)

@app.route('/delete-insert/<int:insert_id>')
def delete_insert(insert_id):
    """Delete an insert and its pages."""
    try:
        # Get the insert to verify it exists
        inserts = db_manager.get_inserts()
        insert = next((i for i in inserts if i['id'] == insert_id), None)
        
        if not insert:
            flash('Insert not found', 'danger')
            return redirect(url_for('view_inserts'))
        
        # First, we need to implement the delete_insert method in DatabaseManager
        if db_manager.delete_insert(insert_id):
            # Try to delete the file from disk if it exists
            if os.path.exists(insert['file_path']):
                try:
                    os.remove(insert['file_path'])
                    flash(f'Insert "{insert["name"]}" and file deleted successfully', 'success')
                except:
                    # If file delete fails, just report the database record was deleted
                    flash(f'Insert "{insert["name"]}" deleted from database (file may remain on disk)', 'success')
            else:
                flash(f'Insert "{insert["name"]}" deleted successfully', 'success')
        else:
            flash('Failed to delete insert', 'danger')
            
    except Exception as e:
        logger.error(f"Error deleting insert: {str(e)}", exc_info=True)
        flash(f'Error deleting insert: {str(e)}', 'danger')
        
    return redirect(url_for('view_inserts'))

@app.route('/view-insert/<int:insert_id>')
def view_insert(insert_id):
    """View an insert with its extracted pages."""
    # Get the insert
    inserts = db_manager.get_inserts()
    insert = next((i for i in inserts if i['id'] == insert_id), None)
    
    if not insert:
        flash('Insert not found', 'danger')
        return redirect(url_for('view_inserts'))
    
    # Get the insert pages
    pages = db_manager.get_insert_pages(insert_id)
    
    # Determine file type for proper rendering
    file_type = insert['filename'].split('.')[-1].lower()
    
    # Get file size in formatted string (e.g., "123 KB")
    file_size_formatted = "Unknown"
    try:
        if os.path.exists(insert['file_path']):
            file_size = os.path.getsize(insert['file_path'])
            # Format file size
            if file_size < 1024:
                file_size_formatted = f"{file_size} B"
            elif file_size < 1024 * 1024:
                file_size_formatted = f"{file_size / 1024:.1f} KB"
            else:
                file_size_formatted = f"{file_size / (1024 * 1024):.1f} MB"
    except Exception as e:
        logger.error(f"Error getting file size for insert {insert_id}: {str(e)}")
    
    # Get usage statistics for this insert (if available)
    usage_stats = None
    try:
        # This is a placeholder - you would implement this based on your data model
        # For example, you might query the database for documents that match this insert
        matches = []  # You would get this from your database
        
        if matches:
            # Calculate usage statistics
            usage_stats = {
                'document_count': len(matches),
                'avg_match_score': sum(match.get('match_score', 0) * 100 for match in matches) / len(matches),
                'last_match_date': max(match.get('match_date', '') for match in matches if match.get('match_date'))
            }
    except Exception as e:
        logger.error(f"Error getting usage statistics for insert {insert_id}: {str(e)}")
    
    return render_template(
        'insert_view.html',
        insert=insert,
        pages=pages,
        file_type=file_type,
        file_size_formatted=file_size_formatted,
        usage_stats=usage_stats
    )

@app.route('/serve-insert/<int:insert_id>')
def serve_insert(insert_id):
    """Serve the insert file for viewing."""
    # Get the insert
    inserts = db_manager.get_inserts()
    insert = next((i for i in inserts if i['id'] == insert_id), None)
    
    if not insert or not os.path.exists(insert['file_path']):
        flash('Insert file not found', 'danger')
        return redirect(url_for('view_inserts'))
    
    return send_file(
        insert['file_path'],
        as_attachment=False,
        download_name=insert['filename']
    )

@app.route('/api/document-statistics')
def api_document_statistics():
    """API endpoint for document statistics."""
    try:
        stats = db_manager.get_document_statistics()
        return jsonify(stats)
    except Exception as e:
        app.logger.error(f"Error in document statistics API: {str(e)}")
        # Return a minimal response with error info
        return jsonify({
            'error': str(e),
            'total_documents': 0,
            'total_paragraphs': 0,
            'duplicates': 0,
            'unique_paragraphs': 0
        })

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
    logger.error(f"Internal server error: {str(e)}", exc_info=True)
    return render_template('500.html'), 500

if __name__ == '__main__':
    # NOTE: Using socketio.run instead of app.run is critical for Socket.IO to work
    socketio.run(app, debug=True)