/**
 * Simplified Document Upload Handler
 * Focus only on document counter
 */

// Global socket variable for communication
let socket;

document.addEventListener('DOMContentLoaded', function() {
    console.log('Document upload script loaded');
    
    // Get form elements
    const uploadForm = document.getElementById('uploadForm');
    const uploadButton = document.getElementById('uploadButton');
    
    // Get counter elements - these are what we care about
    const progressContainer = document.getElementById('uploadProgress');
    const completedFilesElement = document.getElementById('completedFiles');
    const totalFilesElement = document.getElementById('totalFiles');
    const currentFileElement = document.getElementById('currentFileUploading');
    
    // Log if elements are found
    console.log('Upload form found:', !!uploadForm);
    console.log('Progress container found:', !!progressContainer);
    console.log('Completed files counter found:', !!completedFilesElement);
    console.log('Total files counter found:', !!totalFilesElement);
    
    // Initialize Socket.IO for real-time counting
    try {
        if (typeof io !== 'undefined') {
            console.log('Initializing Socket.IO...');
            socket = io();
            
            socket.on('connect', function() {
                console.log('Socket.IO connected successfully');
            });
            
            socket.on('connect_error', function(error) {
                console.error('Socket.IO connection error:', error);
            });
            
            // Handle document counter updates
            socket.on('upload_progress', function(data) {
                console.log('Received progress:', data);
                updateDocumentCounter(data);
            });
            
            // Handle completion
            socket.on('upload_complete', function(data) {
                console.log('Upload complete:', data);
                showCompletionMessage(data);
            });
        } else {
            console.warn('Socket.IO library not found. Real-time updates unavailable.');
        }
    } catch (e) {
        console.error('Error initializing Socket.IO:', e);
    }
    
    // Update document counter elements
    function updateDocumentCounter(data) {
        try {
            // Update document count
            if (completedFilesElement) {
                completedFilesElement.textContent = data.completed || '0';
            }
            
            if (totalFilesElement) {
                totalFilesElement.textContent = data.total || '0';
            }
            
            // Update current file name if available
            if (currentFileElement && data.current_file) {
                currentFileElement.textContent = data.current_file;
            }
            
            // Show progress container
            if (progressContainer) {
                progressContainer.style.display = 'block';
            }
        } catch (error) {
            console.error('Error updating document counter:', error);
        }
    }
    
    // Show completion notification
    function showCompletionMessage(data) {
        try {
            // Create toast notification
            const toast = document.createElement('div');
            toast.className = 'position-fixed bottom-0 end-0 p-3';
            toast.style.zIndex = '11';
            toast.innerHTML = `
                <div class="toast show" role="alert" aria-live="assertive" aria-atomic="true">
                    <div class="toast-header bg-success text-white">
                        <i class="bi bi-check-circle me-2"></i>
                        <strong class="me-auto">Upload Complete</strong>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
                    </div>
                    <div class="toast-body">
                        Successfully processed ${data.processed} out of ${data.total} documents.
                    </div>
                </div>
            `;
            document.body.appendChild(toast);
            
            // Remove after 5 seconds
            setTimeout(function() {
                toast.remove();
            }, 5000);
            
            // Reset button
            if (uploadButton) {
                uploadButton.disabled = false;
                uploadButton.innerHTML = '<i class="bi bi-cloud-upload me-1"></i>Upload';
            }
            
            // Refresh page after delay
            setTimeout(() => {
                window.location.href = '/';
            }, 3000);
        } catch (error) {
            console.error('Error showing completion:', error);
        }
    }
    
    // Handle form submission
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            console.log('Upload form submitted');
            
            const files = document.getElementById('files').files;
            if (files.length === 0) {
                alert('Please select at least one file');
                return;
            }
            
            console.log(`Uploading ${files.length} files`);
            
            // Update total files count immediately
            if (totalFilesElement) {
                totalFilesElement.textContent = files.length;
            }
            
            // Reset completed files count
            if (completedFilesElement) {
                completedFilesElement.textContent = '0';
            }
            
            // Show progress container
            if (progressContainer) {
                progressContainer.style.display = 'block';
            }
            
            // Disable button during upload
            if (uploadButton) {
                uploadButton.disabled = true;
                uploadButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Uploading...';
            }
            
            // Set current file text
            if (currentFileElement) {
                currentFileElement.textContent = 'Preparing...';
            }
            
            // Create and send form data
            const formData = new FormData(this);
            
            fetch('/upload', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Upload failed: ' + response.statusText);
                }
                return response.json();
            })
            .then(data => {
                console.log('Upload response:', data);
                
                // If socket.io isn't working, use the response to update UI
                if (typeof socket === 'undefined' || !socket.connected) {
                    console.log('Using AJAX response for completion (socket unavailable)');
                    showCompletionMessage({
                        processed: data.processed,
                        total: data.total
                    });
                }
                
                // Refresh or redirect
                if (data.redirect) {
                    window.location.href = data.redirect;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Upload error: ' + error.message);
                
                if (uploadButton) {
                    uploadButton.disabled = false;
                    uploadButton.innerHTML = '<i class="bi bi-cloud-upload me-1"></i>Upload';
                }
                
                if (progressContainer) {
                    progressContainer.style.display = 'none';
                }
            });
        });
    }
    
    // File input change handler for instant count update
    const fileInput = document.getElementById('files');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            // Update total files counter when files are selected
            if (totalFilesElement) {
                totalFilesElement.textContent = this.files.length;
            }
            
            // Reset completed files count
            if (completedFilesElement) {
                completedFilesElement.textContent = '0';
            }
            
            // Also update the file count element if present
            const fileCountEl = document.getElementById('fileCount');
            if (fileCountEl) {
                fileCountEl.textContent = this.files.length + ' files selected';
            }
        });
    }
});