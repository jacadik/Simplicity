// Tags management script
document.addEventListener('DOMContentLoaded', function() {
    // Color picker functionality
    const colorOptions = document.querySelectorAll('.color-option');
    const tagColorInput = document.getElementById('tagColor');
    const colorPreviewContainer = document.querySelector('.color-preview-container');
    const tagNameInput = document.getElementById('tagName');
    const previewTagName = document.getElementById('previewTagName');
    
    // Add active class to the first color option by default
    if (colorOptions.length > 0) {
        colorOptions[0].classList.add('active');
    }
    
    // Update tag name preview when typing
    if (tagNameInput && previewTagName) {
        tagNameInput.addEventListener('input', function() {
            previewTagName.textContent = this.value || 'Sample Tag';
        });
    }
    
    // Handle color selection
    colorOptions.forEach(option => {
        option.addEventListener('click', function() {
            const color = this.getAttribute('data-color');
            
            // Remove active class from all options
            colorOptions.forEach(opt => opt.classList.remove('active'));
            
            // Add active class to the clicked option
            this.classList.add('active');
            
            // Update the hidden input value
            if (tagColorInput) {
                tagColorInput.value = color;
            }
            
            // Update the preview
            if (colorPreviewContainer) {
                colorPreviewContainer.style.backgroundColor = color;
            }
        });
    });
    
    // Search functionality for tags
    const tagSearchInput = document.getElementById('tagSearch');
    if (tagSearchInput) {
        tagSearchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            const tagContainers = document.querySelectorAll('.tag-item-container');
            
            tagContainers.forEach(container => {
                const tagName = container.querySelector('.tag-name').textContent.toLowerCase();
                if (tagName.includes(searchTerm)) {
                    container.style.display = '';
                } else {
                    container.style.display = 'none';
                }
            });
        });
    }
    
    // Add hover effects to tag items
    const tagItems = document.querySelectorAll('.tag-item');
    tagItems.forEach(item => {
        item.addEventListener('mouseenter', function() {
            this.classList.add('shadow');
            this.style.transform = 'translateY(-2px)';
        });
        item.addEventListener('mouseleave', function() {
            this.classList.remove('shadow');
            this.style.transform = 'translateY(0)';
        });
    });
    
    // Add event listeners to delete buttons
    setupDeleteButtons();
    
    // Initialize counter cards animation
    const counterCards = document.querySelectorAll('.counter-card');
    counterCards.forEach((card, index) => {
        // Stagger animation delay
        setTimeout(() => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            
            // Add animation
            card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            
            // Trigger animation
            setTimeout(() => {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, 50);
        }, index * 100);
    });
});

function setupDeleteButtons() {
    const deleteButtons = document.querySelectorAll('.delete-tag-btn');
    deleteButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tagId = this.getAttribute('data-tag-id');
            const tagName = this.getAttribute('data-tag-name');
            deleteTag(tagId, tagName, this);
        });
    });
}

function deleteTag(tagId, tagName, element) {
    if (confirm(`Are you sure you want to delete the tag "${tagName}"? This will remove it from all paragraphs.`)) {
        // Show deletion in progress
        const tagItem = document.getElementById(`tag-${tagId}`);
        if (tagItem) {
            tagItem.style.opacity = '0.5';
            tagItem.style.pointerEvents = 'none';
        }
        
        // Ensure jQuery is available
        if (typeof $ === 'undefined') {
            console.error('jQuery is not available');
            return;
        }
        
        $.ajax({
            url: '/delete-tag',
            method: 'POST',
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            },
            data: {
                tag_id: tagId
            },
            success: function(response) {
                if (response && response.success) {
                    // Create success toast notification
                    const toast = document.createElement('div');
                    toast.className = 'position-fixed bottom-0 end-0 p-3';
                    toast.style.zIndex = '11';
                    toast.innerHTML = `
                        <div class="toast show" role="alert" aria-live="assertive" aria-atomic="true">
                            <div class="toast-header bg-success text-white">
                                <i class="bi bi-check-circle me-2"></i>
                                <strong class="me-auto">Success</strong>
                                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
                            </div>
                            <div class="toast-body">
                                Tag "${tagName}" deleted successfully
                            </div>
                        </div>
                    `;
                    document.body.appendChild(toast);
                    
                    // Remove the tag element from DOM with animation
                    const tagContainer = $(element).closest('.tag-item-container');
                    tagContainer.fadeOut(300, function() { 
                        $(this).remove(); 
                        
                        // Check if any tags are left after this removal
                        const remainingTags = document.querySelectorAll('.tag-item');
                        if (remainingTags.length === 0) {
                            const cardBody = document.querySelector('.card-body');
                            if (cardBody) {
                                cardBody.innerHTML = `
                                    <div class="text-center py-5">
                                        <div class="empty-state">
                                            <i class="bi bi-tags text-muted mb-3" style="font-size: 4rem;"></i>
                                            <h5>No tags created yet</h5>
                                            <p class="text-muted">Create tags to organize and categorize your paragraphs</p>
                                        </div>
                                    </div>
                                `;
                            }
                        }
                    });
                    
                    // Auto-dismiss the toast after 3 seconds
                    setTimeout(function() {
                        toast.remove();
                    }, 3000);
                } else {
                    alert('Failed to delete tag: ' + (response && response.message ? response.message : 'Unknown error'));
                    
                    // Restore the tag item
                    if (tagItem) {
                        tagItem.style.opacity = '1';
                        tagItem.style.pointerEvents = 'auto';
                    }
                }
            },
            error: function(xhr, status, error) {
                alert('An error occurred while deleting the tag: ' + error);
                
                // Restore the tag item
                if (tagItem) {
                    tagItem.style.opacity = '1';
                    tagItem.style.pointerEvents = 'auto';
                }
            }
        });
    }
}