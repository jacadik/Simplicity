/**
 * Toast Notification System
 * A modern, consistent notification system for the Document Analyzer app
 */

class NotificationSystem {
  constructor() {
    this.defaultDuration = 3000;
    this.container = null;
    this.createContainer();
  }

  /**
   * Create the container for all notifications
   */
  createContainer() {
    // Check if container already exists
    if (document.getElementById('toast-container')) {
      this.container = document.getElementById('toast-container');
      return;
    }
    
    // Create a new container
    this.container = document.createElement('div');
    this.container.id = 'toast-container';
    this.container.className = 'position-fixed bottom-0 end-0 p-3';
    this.container.style.zIndex = '1060';
    document.body.appendChild(this.container);
  }

  /**
   * Show a notification
   * @param {string} message - The message to display
   * @param {string} type - Type of notification: 'success', 'danger', 'warning', 'info'
   * @param {number} duration - Duration in milliseconds
   */
  show(message, type = 'success', duration = this.defaultDuration) {
    // Create new notification element
    const toast = document.createElement('div');
    const toastId = 'toast-' + Date.now();
    toast.id = toastId;
    toast.className = 'toast-notification shadow-lg rounded-lg overflow-hidden mb-2';
    toast.style.backgroundColor = 'white';
    toast.style.maxWidth = '350px';
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(20px)';
    toast.style.transition = 'all 0.3s ease';
    
    // Get color based on type
    const color = this.getColorForType(type);
    toast.style.borderLeft = `4px solid ${color}`;
    
    // Get icon for type
    const icon = this.getIconForType(type);
    
    // Create content
    toast.innerHTML = `
      <div class="d-flex align-items-center p-3">
        <div class="me-3 rounded-circle d-flex align-items-center justify-content-center" 
             style="background-color: ${color}; width: 36px; height: 36px; flex-shrink: 0;">
          <i class="bi ${icon} text-white"></i>
        </div>
        <div class="flex-grow-1">
          <div class="fw-medium" style="color: #4a4a49;">${message}</div>
        </div>
        <button type="button" class="btn-close ms-2" style="font-size: 0.75rem;" 
                onclick="document.getElementById('${toastId}').remove()" 
                aria-label="Close"></button>
      </div>
    `;
    
    // Add to container
    this.container.appendChild(toast);
    
    // Show with animation
    setTimeout(() => {
      toast.style.opacity = '1';
      toast.style.transform = 'translateY(0)';
    }, 10);
    
    // Auto-remove after duration
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(20px)';
      
      // Remove from DOM after animation completes
      setTimeout(() => {
        if (document.getElementById(toastId)) {
          document.getElementById(toastId).remove();
        }
      }, 300);
    }, duration);
    
    return toastId;
  }
  
  /**
   * Show a success notification
   * @param {string} message - The message to display
   * @param {number} duration - Duration in milliseconds
   */
  success(message, duration = this.defaultDuration) {
    return this.show(message, 'success', duration);
  }
  
  /**
   * Show an error notification
   * @param {string} message - The message to display
   * @param {number} duration - Duration in milliseconds
   */
  error(message, duration = this.defaultDuration) {
    return this.show(message, 'danger', duration);
  }
  
  /**
   * Show a warning notification
   * @param {string} message - The message to display
   * @param {number} duration - Duration in milliseconds
   */
  warning(message, duration = this.defaultDuration) {
    return this.show(message, 'warning', duration);
  }
  
  /**
   * Show an info notification
   * @param {string} message - The message to display
   * @param {number} duration - Duration in milliseconds
   */
  info(message, duration = this.defaultDuration) {
    return this.show(message, 'info', duration);
  }
  
  /**
   * Get the appropriate color for the notification type
   * @param {string} type - Type of notification
   * @returns {string} CSS color value
   */
  getColorForType(type) {
    switch (type) {
      case 'success': return '#4fccc4'; // var(--success-color)
      case 'danger': return '#fd565c';  // var(--danger-color)
      case 'warning': return '#ffc107'; // var(--warning-color)
      case 'info': default: return '#5787eb'; // var(--primary-color)
    }
  }
  
  /**
   * Get the appropriate icon for the notification type
   * @param {string} type - Type of notification
   * @returns {string} Bootstrap icon class
   */
  getIconForType(type) {
    switch (type) {
      case 'success': return 'bi-check-circle-fill';
      case 'danger': return 'bi-exclamation-circle-fill';
      case 'warning': return 'bi-exclamation-triangle-fill';
      case 'info': default: return 'bi-info-circle-fill';
    }
  }
}

// Create global instance
const notifier = new NotificationSystem();

// Add to window object for global access
window.notifier = notifier;

// Example usage:
// notifier.success('Document uploaded successfully');
// notifier.error('Failed to upload document');
// notifier.warning('File size exceeds recommended limit');
// notifier.info('Processing document in background');
