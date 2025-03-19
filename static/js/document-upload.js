/**
 * Document Batch Upload Handler
 * Handles large batch uploads with real-time progress tracking
 */

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const uploadButton = document.getElementById('uploadButton');
    const progressBar = document.querySelector('.progress-bar');
    const progressContainer = document.getElementById('uploadProgress');
    const currentFileElement = document.getElementById('currentFileUploading');
    const completedFilesElement = document.getElementById('completedFiles');
    const statsContainer = document.getElementById('uploadStats');
    
    // Socket.IO initialization for real-time progress updates
    let socket;
    try {
        // Only initialize if the socket.io library is available
        if (typeof io !== 'undefined') {
            socket = io();
            
            // Handle progress updates
            socket.on('upload_progress', function(data) {
                updateProgressUI(data);
            });
            
            // Handle batch completion
            socket.on('upload_complete', function(data) {
                showCompletionMessage(data);
            });
        }
    } catch (e) {
        console.warn('Socket.IO not available, progress updates will not be real-time', e);
    }
    
    // Update progress UI with data from server
    function updateProgressUI(data) {
        // Update progress bar
        if (progressBar) {
            progressBar.style.width = data.progress_percent + '%';
            progressBar.setAttribute('aria-valuenow', data.progress_percent);
            progressBar.textContent = data.progress_percent + '%';
        }
        
        // Update current file and completed count
        if (currentFileElement) {
            currentFileElement.textContent = data.current_file;
        }
        
        if (completedFilesElement) {
            completedFilesElement.textContent = data.completed + ' / ' + data.total;
        }
        
        // Update stats if available
        if (statsContainer && data.stats) {
            const successRate = ((data.stats.success_count / data.stats.total) * 100).toFixed(1);
            statsContainer.innerHTML = `
                <div class="d-flex justify-content-between">
                    <div>Success rate: <strong>${successRate}%</strong></div>
                    <div>Avg. processing time: <strong>${data.stats.avg_time.toFixed(2)}s</strong></div>
                </div>
            `;
        }
        
        // Show progress container if hidden
        if (progressContainer) {
            progressContainer.style.display = 'block';
        }
    }
    
    // Show completion message when batch is done
    function showCompletionMessage(data) {
        // Create success toast notification
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
                    Successfully processed ${data.processed} out of ${data.total} documents in ${data.elapsed_time.toFixed(1)} seconds.
                </div>
            </div>
        `;
        document.body.appendChild(toast);
        
        // Auto-dismiss the toast after 5 seconds
        setTimeout(function() {
            toast.remove();
        }, 5000);
        
        // Update UI
        if (uploadButton) {
            uploadButton.disabled = false;
            uploadButton.innerHTML = '<i class="bi bi-cloud-upload me-1"></i>Upload';
        }
        
        // Update progress to 100%
        if (progressBar) {
            progressBar.style.width = '100%';
            progressBar.setAttribute('aria-valuenow', 100);
            progressBar.textContent = '100%';
        }
        
        // Show completion
        if (currentFileElement) {
            currentFileElement.textContent = 'Complete!';
        }
    }
    
    // Handle form submission for large batch uploads
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const files = document.getElementById('files').files;
            if (files.length === 0) {
                alert('Please select at least one file');
                return;
            }
            
            // Show file count warning for large batches
            if (files.length > 50) {
                if (!confirm(`You are about to upload ${files.length} files. This may take some time. Continue?`)) {
                    return;
                }
            }
            
            // Create FormData object
            const formData = new FormData(this);
            
            // Show progress bar and disable submit button
            if (progressContainer) {
                progressContainer.style.display = 'block';
            }
            
            if (uploadButton) {
                uploadButton.disabled = true;
                uploadButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Uploading...';
            }
            
            // Reset progress indicators
            if (progressBar) {
                progressBar.style.width = '0%';
                progressBar.setAttribute('aria-valuenow', 0);
                progressBar.textContent = '0%';
            }
            
            if (currentFileElement) {
                currentFileElement.textContent = 'Preparing...';
            }
            
            if (completedFilesElement) {
                completedFilesElement.textContent = '0 / ' + files.length;
            }
            
            // Send AJAX request
            fetch('/upload', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok: ' + response.statusText);
                }
                return response.json();
            })
            .then(data => {
                // If no Socket.IO is available, show completion based on AJAX response
                if (!socket) {
                    showCompletionMessage({
                        processed: data.processed,
                        total: data.total,
                        elapsed_time: data.elapsed_time || 0
                    });
                }
                
                // If requested by server or after 5 seconds, redirect to refresh the page
                if (data.redirect) {
                    window.location.href = data.redirect;
                } else {
                    setTimeout(() => {
                        window.location.href = '/';
                    }, 5000);
                }
            })
            .catch(error => {
                // On error, show alert and reset form
                console.error('Error uploading files:', error);
                alert('Error uploading files: ' + error.message);
                
                if (progressContainer) {
                    progressContainer.style.display = 'none';
                }
                
                if (uploadButton) {
                    uploadButton.disabled = false;
                    uploadButton.innerHTML = '<i class="bi bi-cloud-upload me-1"></i>Upload';
                }
            });
        });
    }
    
    // Add drag and drop support for files
    const dropZone = document.getElementById('dropZone');
    if (dropZone) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            dropZone.classList.add('drag-highlight');
        }
        
        function unhighlight() {
            dropZone.classList.remove('drag-highlight');
        }
        
        dropZone.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            const fileInput = document.getElementById('files');
            
            if (fileInput) {
                fileInput.files = files;
                // Update file count display
                const fileCountEl = document.getElementById('fileCount');
                if (fileCountEl) {
                    fileCountEl.textContent = files.length + ' files selected';
                }
            }
        }
    }
    
    // Show file count when files are selected
    const fileInput = document.getElementById('files');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const fileCountEl = document.getElementById('fileCount');
            if (fileCountEl) {
                fileCountEl.textContent = this.files.length + ' files selected';
            }
        });
    }
});