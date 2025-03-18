/**
 * Document Analyzer
 * Utility functions for shared functionality
 */

/**
 * Format an ISO-8601 date string to display format
 * @param {string} isoDateString - ISO date string like "2023-01-15T12:30:45.000Z"
 * @param {boolean} includeTime - Whether to include time in the result
 * @return {string} Formatted date string like "15-01-2023" or "15-01-2023 12:30"
 */
function formatDate(isoDateString, includeTime = false) {
    if (!isoDateString) return '-';
    
    try {
        const parts = isoDateString.split('T');
        const datePart = parts[0];
        let timePart = '';
        
        if (includeTime && parts.length > 1) {
            timePart = ' ' + parts[1].split('.')[0].substring(0, 5);
        }
        
        // Convert from YYYY-MM-DD to DD-MM-YYYY
        const dateParts = datePart.split('-');
        if (dateParts.length === 3) {
            return dateParts[2] + '-' + dateParts[1] + '-' + dateParts[0] + timePart;
        }
        
        return datePart + timePart;
    } catch (e) {
        console.error('Error formatting date:', e);
        return isoDateString || '-';
    }
}

/**
 * Check if jQuery is available
 * @returns {boolean} True if jQuery is available
 */
function isJQueryAvailable() {
    return typeof $ !== 'undefined' && $ !== null && typeof $.ajax === 'function';
}

/**
 * Create a toast notification
 * @param {string} message - Toast message
 * @param {string} type - Toast type (success, danger, warning, info)
 * @param {number} duration - Duration in milliseconds
 */
function showToast(message, type = 'success', duration = 3000) {
    const toast = document.createElement('div');
    toast.className = 'position-fixed bottom-0 end-0 p-3';
    toast.style.zIndex = '11';
    
    const bgClass = type === 'success' ? 'bg-success' : 
                    type === 'danger' ? 'bg-danger' :
                    type === 'warning' ? 'bg-warning' : 'bg-info';
    
    const icon = type === 'success' ? 'check-circle' : 
                type === 'danger' ? 'exclamation-circle' :
                type === 'warning' ? 'exclamation-triangle' : 'info-circle';
    
    toast.innerHTML = `
        <div class="toast show" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header ${bgClass} text-white">
                <i class="bi bi-${icon} me-2"></i>
                <strong class="me-auto">${type.charAt(0).toUpperCase() + type.slice(1)}</strong>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;
    
    document.body.appendChild(toast);
    
    // Auto-dismiss the toast after duration
    setTimeout(function() {
        toast.remove();
    }, duration);
}

/**
 * Safe text truncation with proper ellipsis
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length before truncation
 * @returns {string} Truncated text
 */
function truncateText(text, maxLength = 100) {
    if (!text || text.length <= maxLength) {
        return text || '';
    }
    
    // Find a word boundary to truncate at
    const boundary = text.lastIndexOf(' ', maxLength);
    const truncated = boundary > 0 ? text.substring(0, boundary) : text.substring(0, maxLength);
    
    return truncated + '...';
}

/**
 * Highlight search terms in text
 * @param {HTMLElement} element - Element containing text to highlight
 * @param {string} term - Search term to highlight
 */
function highlightSearchTerm(element, term) {
    if (!element || !term || term.trim() === '') {
        return;
    }
    
    const originalText = element.innerHTML;
    // Remove any existing highlights
    const cleanText = originalText.replace(/<mark class="search-highlight">([^<]+)<\/mark>/g, '$1');
    
    // Add new highlights
    const regex = new RegExp(term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
    const highlightedText = cleanText.replace(regex, '<mark class="search-highlight">$&</mark>');
    
    element.innerHTML = highlightedText;
}

/**
 * Apply a debounce function to avoid excessive function calls
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
function debounce(func, wait = 300) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Make functions available globally
window.formatDate = formatDate;
window.isJQueryAvailable = isJQueryAvailable;
window.showToast = showToast;
window.truncateText = truncateText;
window.highlightSearchTerm = highlightSearchTerm;
window.debounce = debounce;
