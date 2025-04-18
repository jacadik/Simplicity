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
from utils.database.models import Document, Paragraph, Tag, SimilarityResult, Cluster, cluster_paragraphs, paragraph_tags
from utils.database.manager import DatabaseManager

# New imports for enhanced query capabilities
from sqlalchemy import func, or_, and_, text, distinct, case
from sqlalchemy.sql.expression import cast
from sqlalchemy.types import String

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
    
    # Get documents and tags for filters
    documents = db_manager.get_documents()
    tags = db_manager.get_tags()
    
    # Create a dictionary to map filenames to document IDs for easy lookup in the template
    filename_to_doc_id = {doc['filename']: doc['id'] for doc in documents}
    
    # Get initial statistics to show before paragraphs are loaded
    try:
        session = db_manager.Session()
        total_paragraphs = session.query(Paragraph).count()
        session.close()
    except Exception as e:
        app.logger.error(f"Error getting paragraph count: {str(e)}")
        total_paragraphs = 0
    
    return render_template(
        'paragraphs.html', 
        documents=documents, 
        tags=tags, 
        selected_document=document_id,
        show_all_duplicates=show_all_duplicates,
        filename_to_doc_id=filename_to_doc_id,
        total_paragraphs=total_paragraphs,
        # Do not pass paragraphs - they'll be loaded via AJAX
    )

# New API endpoints for optimized paragraph rendering
# Fix for api_get_paragraphs function in app.py

@app.route('/api/paragraphs')
def api_get_paragraphs():
    """API endpoint for paginated paragraphs with filtering options."""
    try:
        # Parse pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 25, type=int)
        
        # Limit per_page to reasonable values
        per_page = min(100, max(10, per_page))
        
        # Parse filter parameters
        document_id = request.args.get('document_id', type=int)
        paragraph_type = request.args.get('type')
        tag_id = request.args.get('tag_id', type=int)
        min_length = request.args.get('min_length', 0, type=int)
        search_query = request.args.get('search')
        
        # FIXED: Always default to collapsed (0) unless explicitly set to 1
        # Ensure parameter is explicitly read as string and compared
        show_all_duplicates = request.args.get('show_all_duplicates', '0') == '1'
        
        # Add debug logging
        app.logger.info(f"API request with params: show_all_duplicates={show_all_duplicates}, document_id={document_id}")
        
        # Parse sorting parameters
        sort_by = request.args.get('sort_by', 'position')
        sort_direction = request.args.get('sort_direction', 'asc')
        
        # Create a database session
        session = db_manager.Session()
        
        # Build the base query
        query = session.query(
            Paragraph,
            Document.filename,
            Document.upload_date
        ).join(
            Document,
            Paragraph.document_id == Document.id
        )
        
        # Apply filters
        if document_id:
            query = query.filter(Paragraph.document_id == document_id)
        
        if paragraph_type:
            query = query.filter(Paragraph.paragraph_type == paragraph_type)
        
        if tag_id:
            query = query.join(
                paragraph_tags,
                Paragraph.id == paragraph_tags.c.paragraph_id
            ).filter(paragraph_tags.c.tag_id == tag_id)
        
        if min_length > 0:
            query = query.filter(func.length(Paragraph.content) >= min_length)
        
        if search_query:
            search_term = f"%{search_query}%"
            query = query.filter(Paragraph.content.ilike(search_term))
        
        # Apply sorting
        if sort_by == 'position':
            if sort_direction == 'asc':
                query = query.order_by(Paragraph.position.asc())
            else:
                query = query.order_by(Paragraph.position.desc())
        elif sort_by == 'length':
            if sort_direction == 'asc':
                query = query.order_by(func.length(Paragraph.content).asc())
            else:
                query = query.order_by(func.length(Paragraph.content).desc())
        elif sort_by == 'document':
            if sort_direction == 'asc':
                query = query.order_by(Document.filename.asc())
            else:
                query = query.order_by(Document.filename.desc())
        
        # Special case for sorting by occurrences
        duplicate_counts = None
        if sort_by == 'occurrences':
            # Create a subquery to count occurrences of each content
            duplicate_counts = session.query(
                Paragraph.content,
                func.count(Paragraph.id).label('occurrence_count')
            ).group_by(Paragraph.content).subquery()
            
            query = query.outerjoin(
                duplicate_counts,
                Paragraph.content == duplicate_counts.c.content
            )
            
            if sort_direction == 'asc':
                query = query.order_by(
                    func.coalesce(duplicate_counts.c.occurrence_count, 1).asc()
                )
            else:
                query = query.order_by(
                    func.coalesce(duplicate_counts.c.occurrence_count, 1).desc()
                )
        
        # FIXED: Handle duplicate collapsing if requested, even when document_id is specified
        if not show_all_duplicates:
            app.logger.info("Collapsing duplicates in query")
            
            # First, find content that appears multiple times
            duplicate_content_query = session.query(
                Paragraph.content
            ).group_by(
                Paragraph.content
            ).having(
                func.count() > 1
            ).subquery()
            
            # Only show the first occurrence of each duplicate content
            unique_paragraphs_query = session.query(
                func.min(Paragraph.id).label('min_id')
            ).join(
                duplicate_content_query,
                Paragraph.content == duplicate_content_query.c.content
            ).group_by(
                Paragraph.content
            ).subquery()
            
            # Now get all unique content paragraphs and first instances of duplicates
            query = query.filter(
                or_(
                    Paragraph.id.in_(session.query(unique_paragraphs_query.c.min_id)),
                    ~Paragraph.content.in_(session.query(duplicate_content_query.c.content))
                )
            )
        
        # Count total filtered records (before pagination)
        total_count = query.count()
        
        # Calculate total pages
        total_pages = (total_count + per_page - 1) // per_page
        
        # Ensure page is within valid range
        if page < 1:
            page = 1
        elif page > total_pages and total_pages > 0:
            page = total_pages
        
        # Apply pagination
        query = query.offset((page - 1) * per_page).limit(per_page)
        
        # Execute query
        results = query.all()
        
        # Process results
        paragraphs_data = []
        for para, filename, upload_date in results:
            # Get tags for this paragraph
            tags = []
            for tag in para.tags:
                tags.append({
                    'id': tag.id,
                    'name': tag.name,
                    'color': tag.color
                })
            
            # FIXED: Get document references (where this paragraph appears) regardless of document_id filter
            # This ensures we always show all related documents for duplicate paragraphs
            if not show_all_duplicates:
                # Find all paragraphs with the same content
                same_content_paras = session.query(
                    Paragraph.id,
                    Paragraph.document_id,
                    Document.filename
                ).join(
                    Document,
                    Paragraph.document_id == Document.id
                ).filter(
                    Paragraph.content == para.content
                ).all()
                
                doc_refs = []
                for _, doc_id, doc_filename in same_content_paras:
                    doc_refs.append({
                        'id': doc_id,
                        'filename': doc_filename
                    })
                
                appears_in_multiple = len(doc_refs) > 1
            else:
                doc_refs = [{
                    'id': para.document_id,
                    'filename': filename
                }]
                appears_in_multiple = False
            
            # Word count (to avoid recalculating in JavaScript)
            word_count = len(para.content.split()) if para.content else 0
            
            paragraphs_data.append({
                'id': para.id,
                'content': para.content,
                'documentId': para.document_id,
                'documentName': filename,
                'type': para.paragraph_type,
                'position': para.position,
                'headerContent': para.header_content,
                'contentLength': len(para.content) if para.content else 0,
                'wordCount': word_count,
                'tags': tags,
                'documentReferences': doc_refs,
                'occurrences': len(doc_refs),
                'appearsInMultiple': appears_in_multiple
            })
        
        # Close the session
        session.close()
        
        # Return JSON response
        return jsonify({
            'paragraphs': paragraphs_data,
            'current_page': page,
            'per_page': per_page,
            'total_pages': total_pages,
            'total_items': total_count
        })
        
    except Exception as e:
        app.logger.error(f"API error retrieving paragraphs: {str(e)}", exc_info=True)
        
        # Return error response
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'paragraphs': [],
            'current_page': 1,
            'per_page': per_page,
            'total_pages': 0,
            'total_items': 0
        }), 500

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

# This goes at the very end of your app.py file
# After all other routes, functions, and code

# Run the application with standard Flask method
if __name__ == "__main__":
    print("Starting the Flask server on http://0.0.0.0:5000...")
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()