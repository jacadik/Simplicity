/**
 * Document Batch Upload Handler
 * Handles large batch uploads with real-time progress tracking
 * Version: 3.0 with enhanced counters
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Document upload script loaded');
    
    // Set a flag to notify base.html that the upload handler is initialized
    window.documentUploadInitialized = true;
    
    // Get form elements
    const uploadForm = document.getElementById('uploadForm');
    const folderForm = document.getElementById('folderForm');
    const uploadButton = document.getElementById('uploadButton');
    const importButton = document.getElementById('importButton');
    
    // File upload progress elements
    const progressBar = document.querySelector('#uploadProgress .progress-bar');
    const progressContainer = document.getElementById('uploadProgress');
    const currentFileElement = document.getElementById('currentFileUploading');
    const completedFilesElement = document.getElementById('completedFiles');
    const totalFilesElement = document.getElementById('totalFiles');
    const completedFilesCountElement = document.getElementById('completedFilesCount');
    const remainingFilesCountElement = document.getElementById('remainingFilesCount');
    const avgProcessingTimeElement = document.getElementById('avgProcessingTime');
    const estimatedTimeRemainingElement = document.getElementById('estimatedTimeRemaining');
    const statsContainer = document.getElementById('uploadStats');
    
    // Folder upload progress elements
    const folderProgressBar = document.querySelector('#folderProgress .progress-bar');
    const folderProgressContainer = document.getElementById('folderProgress');
    const currentFolderFileElement = document.getElementById('currentFolderFileUploading');
    const completedFolderFilesElement = document.getElementById('completedFolderFiles');
    const totalFolderFilesElement = document.getElementById('totalFolderFiles');
    const completedFolderFilesCountElement = document.getElementById('completedFolderFilesCount');
    const remainingFolderFilesCountElement = document.getElementById('remainingFolderFilesCount');
    const avgFolderProcessingTimeElement = document.getElementById('avgFolderProcessingTime');
    const estimatedFolderTimeRemainingElement = document.getElementById('estimatedFolderTimeRemaining');
    const folderStatsContainer = document.getElementById('folderStats');
    
    // Upload process counter elements
    const uploadCounterElement = document.getElementById('uploadCounter');
    const uploadCounterContainer = document.getElementById('uploadCounterContainer');
    
    console.log('Upload form found:', !!uploadForm);
    console.log('Folder form found:', !!folderForm);
    console.log('Progress container found:', !!progressContainer);
    console.log('Folder progress container found:', !!folderProgressContainer);
    console.log('Upload counter found:', !!uploadCounterElement);
    
    // Socket.IO initialization for real-time progress updates
    let socket;
    try {
        // Only initialize if the socket.io library is available
        if (typeof io !== 'undefined') {
            console.log('Socket.IO library found, initializing...');
            socket = io();
            
            socket.on('connect', function() {
                console.log('Socket.IO connected successfully');
            });
            
            socket.on('connect_error', function(error) {
                console.error('Socket.IO connection error:', error);
            });
            
            // Handle progress updates for file uploads
            socket.on('upload_progress', function(data) {
                console.log('Received upload_progress event:', data);
                updateProgressUI(data, 'file');
                updateUploadCounter(data.total - data.completed);
            });
            
            // Handle progress updates for folder uploads
            socket.on('folder_upload_progress', function(data) {
                console.log('Received folder_upload_progress event:', data);
                updateProgressUI(data, 'folder');
                updateUploadCounter(data.total - data.completed);
            });
            
            // Handle batch completion
            socket.on('upload_complete', function(data) {
                console.log('Received upload_complete event:', data);
                showCompletionMessage(data, 'file');
                updateUploadCounter(0);
            });
            
            // Handle folder batch completion
            socket.on('folder_upload_complete', function(data) {
                console.log('Received folder_upload_complete event:', data);
                showCompletionMessage(data, 'folder');
                // Ensure counter is reset to zero on completion
                updateUploadCounter(0);
            });
            
            // Handle folder scanning progress
            socket.on('folder_scan_progress', function(data) {
                console.log('Received folder_scan_progress event:', data);
                updateFolderScanProgress(data);
                // Always update the counter with file count during scanning
                updateUploadCounter(data.file_count);
            });
        } else {
            console.warn('Socket.IO library not found. Real-time progress updates will not be available.');
        }
    } catch (e) {
        console.error('Error initializing Socket.IO:', e);
    }
    
    // Format time in human-readable format
    function formatTime(seconds) {
        if (seconds < 60) {
            return `${Math.round(seconds)}s`;
        } else if (seconds < 3600) {
            const minutes = Math.floor(seconds / 60);
            const secs = Math.round(seconds % 60);
            return `${minutes}m ${secs}s`;
        } else {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            return `${hours}h ${minutes}m`;
        }
    }
    
    // Update the upload counter in the header
    function updateUploadCounter(remainingFiles) {
        if (uploadCounterElement) {
            if (remainingFiles > 0) {
                // Update counter text
                uploadCounterElement.textContent = remainingFiles;
                
                if (uploadCounterContainer) {
                    // Make sure counter is visible
                    uploadCounterContainer.style.display = 'inline-flex';
                    
                    // Add pulse animation
                    uploadCounterContainer.classList.add('counter-pulse');
                    setTimeout(() => {
                        uploadCounterContainer.classList.remove('counter-pulse');
                    }, 1000);
                    
                    console.log('Upload counter updated to:', remainingFiles);
                }
            } else {
                // Hide counter when no files remain
                if (uploadCounterContainer) {
                    uploadCounterContainer.style.display = 'none';
                    console.log('Upload counter hidden (zero files)');
                }
            }
        } else {
            console.warn('Upload counter element not found');
        }
    }
    
    // Update progress UI with data from server
    function updateProgressUI(data, type) {
        console.log(`Updating ${type} progress UI:`, data);
        
        // Get the right elements based on upload type
        const elements = type === 'folder' ? {
            progressBar: folderProgressBar,
            currentFile: currentFolderFileElement,
            completedFiles: completedFolderFilesElement,
            totalFiles: totalFolderFilesElement,
            completedFilesCount: completedFolderFilesCountElement,
            remainingFilesCount: remainingFolderFilesCountElement,
            avgProcessingTime: avgFolderProcessingTimeElement,
            estimatedTimeRemaining: estimatedFolderTimeRemainingElement,
            statsContainer: folderStatsContainer,
            progressContainer: folderProgressContainer
        } : {
            progressBar: progressBar,
            currentFile: currentFileElement,
            completedFiles: completedFilesElement,
            totalFiles: totalFilesElement,
            completedFilesCount: completedFilesCountElement,
            remainingFilesCount: remainingFilesCountElement,
            avgProcessingTime: avgProcessingTimeElement,
            estimatedTimeRemaining: estimatedTimeRemainingElement,
            statsContainer: statsContainer,
            progressContainer: progressContainer
        };
        
        try {
            // Update progress bar
            if (elements.progressBar) {
                elements.progressBar.style.width = data.progress_percent + '%';
                elements.progressBar.setAttribute('aria-valuenow', data.progress_percent);
                elements.progressBar.textContent = data.progress_percent + '%';
            }
            
            // Update file counts and info
            if (elements.currentFile) {
                elements.currentFile.textContent = data.current_file || 'Processing...';
            }
            
            if (elements.completedFiles) {
                elements.completedFiles.textContent = data.completed || '0';
            }
            
            if (elements.totalFiles) {
                elements.totalFiles.textContent = data.total || '0';
            }
            
            // Update completed and remaining counts
            if (elements.completedFilesCount) {
                elements.completedFilesCount.textContent = data.completed || '0';
            }
            
            const remaining = data.total - data.completed;
            if (elements.remainingFilesCount) {
                elements.remainingFilesCount.textContent = remaining || '0';
            }
            
            // Calculate and display average processing time and estimated time remaining
            if (data.stats && data.stats.avg_time) {
                if (elements.avgProcessingTime) {
                    elements.avgProcessingTime.textContent = formatTime(data.stats.avg_time);
                }
                
                if (elements.estimatedTimeRemaining) {
                    if (remaining > 0 && data.stats.avg_time > 0) {
                        const estimatedTimeRemaining = remaining * data.stats.avg_time;
                        elements.estimatedTimeRemaining.textContent = formatTime(estimatedTimeRemaining);
                    } else {
                        elements.estimatedTimeRemaining.textContent = 'Almost done...';
                    }
                }
            }
            
            // Update additional stats if available
            if (elements.statsContainer && data.stats) {
                const successRate = ((data.stats.success_count / data.total) * 100).toFixed(1);
                elements.statsContainer.innerHTML = `
                    <div class="d-flex justify-content-between">
                        <div>Success rate: <strong>${successRate}%</strong></div>
                        <div>Progress: <strong>${data.completed}/${data.total}</strong> files</div>
                    </div>
                `;
            }
            
            // Show progress container if hidden
            if (elements.progressContainer) {
                elements.progressContainer.style.display = 'block';
            }
        } catch (error) {
            console.error('Error updating progress UI:', error);
        }
    }
    
    // Update folder scanning progress UI
    function updateFolderScanProgress(data) {
        console.log('Updating folder scan progress:', data);
        try {
            if (currentFolderFileElement) {
                currentFolderFileElement.textContent = `Scanning folder: Found ${data.file_count} files`;
            }
            
            if (totalFolderFilesElement) {
                totalFolderFilesElement.textContent = data.file_count || '0';
            }
            
            if (remainingFolderFilesCountElement) {
                remainingFolderFilesCountElement.textContent = data.file_count || '0';
            }
            
            if (folderProgressBar) {
                // Just show indeterminate progress during scanning
                folderProgressBar.style.width = '100%';
                folderProgressBar.classList.add('progress-bar-animated');
            }
        } catch (error) {
            console.error('Error updating folder scan progress:', error);
        }
    }
    
    // Show completion message when batch is done
    function showCompletionMessage(data, type) {
        console.log(`Showing completion message for ${type}:`, data);
        
        // Get the right elements based on upload type
        const elements = type === 'folder' ? {
            button: importButton,
            progressBar: folderProgressBar,
            currentFile: currentFolderFileElement
        } : {
            button: uploadButton,
            progressBar: progressBar, 
            currentFile: currentFileElement
        };
        
        try {
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
            if (elements.button) {
                elements.button.disabled = false;
                elements.button.innerHTML = type === 'folder' ? 
                    '<i class="bi bi-folder-check me-1"></i>Import' : 
                    '<i class="bi bi-cloud-upload me-1"></i>Upload';
            }
            
            // Update progress to 100%
            if (elements.progressBar) {
                elements.progressBar.style.width = '100%';
                elements.progressBar.setAttribute('aria-valuenow', 100);
                elements.progressBar.textContent = '100%';
                elements.progressBar.classList.remove('progress-bar-animated');
            }
            
            // Show completion
            if (elements.currentFile) {
                elements.currentFile.textContent = 'Complete!';
            }
            
            // Redirect to refresh page after a delay
            setTimeout(() => {
                window.location.href = '/';
            }, 3000);
        } catch (error) {
            console.error('Error showing completion message:', error);
        }
    }
    
    // Handle form submission for file uploads
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
            
            // Update the upload counter with total files
            updateUploadCounter(files.length);
            
            // Show file count warning for large batches
            if (files.length > 50) {
                if (!confirm(`You are about to upload ${files.length} files. This may take some time. Continue?`)) {
                    return;
                }
            }
            
            // Initialize the total file count in the UI
            if (totalFilesElement) {
                totalFilesElement.textContent = files.length;
            }
            
            if (remainingFilesCountElement) {
                remainingFilesCountElement.textContent = files.length;
            }
            
            // Clear any existing stats
            if (statsContainer) {
                statsContainer.innerHTML = '';
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
                progressBar.classList.add('progress-bar-animated');
            }
            
            if (currentFileElement) {
                currentFileElement.textContent = 'Preparing...';
            }
            
            if (completedFilesElement) {
                completedFilesElement.textContent = '0';
            }
            
            if (completedFilesCountElement) {
                completedFilesCountElement.textContent = '0';
            }
            
            if (estimatedTimeRemainingElement) {
                estimatedTimeRemainingElement.textContent = 'Calculating...';
            }
            
            console.log('Sending AJAX request to /upload');
            
            // Send AJAX request
            fetch('/upload', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => {
                console.log('Received response:', response.status);
                if (!response.ok) {
                    throw new Error('Network response was not ok: ' + response.statusText);
                }
                return response.json();
            })
            .then(data => {
                console.log('Upload complete response:', data);
                // If no Socket.IO is available, show completion based on AJAX response
                if (!socket) {
                    showCompletionMessage({
                        processed: data.processed,
                        total: data.total,
                        elapsed_time: data.elapsed_time || 0
                    }, 'file');
                    updateUploadCounter(0);
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
                
                // Reset upload counter
                updateUploadCounter(0);
                console.log('Upload counter reset due to file upload error');
            });
        });
    }
    
    // Handle form submission for folder imports
    if (folderForm) {
        folderForm.addEventListener('submit', function(e) {
            e.preventDefault();
            console.log('Folder form submitted');
            
            const folderPath = document.getElementById('folderPath').value;
            if (!folderPath.trim()) {
                alert('Please enter a valid folder path');
                return;
            }
            
            console.log(`Processing folder: ${folderPath}`);
            
            // Initialize the upload counter to show we're starting a process
            updateUploadCounter(1); // Set to at least 1 to show the counter
            
            // Create FormData object
            const formData = new FormData(this);
            
            // Show progress container and disable submit button
            if (folderProgressContainer) {
                folderProgressContainer.style.display = 'block';
            }
            
            if (importButton) {
                importButton.disabled = true;
                importButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Scanning...';
            }
            
            // Reset progress indicators
            if (folderProgressBar) {
                folderProgressBar.style.width = '0%';
                folderProgressBar.setAttribute('aria-valuenow', 0);
                folderProgressBar.textContent = '0%';
                folderProgressBar.classList.add('progress-bar-animated');
            }
            
            if (currentFolderFileElement) {
                currentFolderFileElement.textContent = 'Scanning folder...';
            }
            
            if (completedFolderFilesElement) {
                completedFolderFilesElement.textContent = '0';
            }
            
            if (totalFolderFilesElement) {
                totalFolderFilesElement.textContent = '...';
            }
            
            if (completedFolderFilesCountElement) {
                completedFolderFilesCountElement.textContent = '0';
            }
            
            if (remainingFolderFilesCountElement) {
                remainingFolderFilesCountElement.textContent = '...';
            }
            
            if (estimatedFolderTimeRemainingElement) {
                estimatedFolderTimeRemainingElement.textContent = 'Calculating...';
            }
            
            console.log('Sending AJAX request to /upload-folder');
            
            // Initialize upload counter with placeholder value until we get real data
            updateUploadCounter(1);  // Start with at least 1 to show counter
            
            // Send AJAX request
            fetch('/upload-folder', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => {
                console.log('Received response:', response.status);
                if (!response.ok) {
                    throw new Error('Network response was not ok: ' + response.statusText);
                }
                return response.json();
            })
            .then(data => {
                console.log('Folder processing complete response:', data);
                // If no Socket.IO is available, show completion based on AJAX response
                if (!socket) {
                    showCompletionMessage({
                        processed: data.processed,
                        total: data.total,
                        elapsed_time: data.elapsed_time || 0
                    }, 'folder');
                    updateUploadCounter(0);
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
                console.error('Error processing folder:', error);
                alert('Error processing folder: ' + error.message);
                
                if (folderProgressContainer) {
                    folderProgressContainer.style.display = 'none';
                }
                
                if (importButton) {
                    importButton.disabled = false;
                    importButton.innerHTML = '<i class="bi bi-folder-check me-1"></i>Import';
                }
                
                // Reset upload counter
                updateUploadCounter(0);
                console.log('Upload counter reset due to folder processing error');
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
                
                // Also update the total files counter in the progress UI
                if (totalFilesElement) {
                    totalFilesElement.textContent = files.length;
                }
                
                if (remainingFilesCountElement) {
                    remainingFilesCountElement.textContent = files.length;
                }
                
                // Update upload counter
                updateUploadCounter(files.length);
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
            
            // Also update the total files counter in the progress UI
            if (totalFilesElement) {
                totalFilesElement.textContent = this.files.length;
            }
            
            if (remainingFilesCountElement) {
                remainingFilesCountElement.textContent = this.files.length;
            }
            
            // Update upload counter
            updateUploadCounter(this.files.length);
        });
    }
});