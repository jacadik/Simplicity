/**
 * Virtualized Paragraphs
 * Optimized rendering for the paragraphs page using virtualization and server-side pagination
 */

console.log("Loading virtualized paragraphs script...");

// Global variables
let allParagraphs = []; // Will store all paragraphs for current page
let filteredParagraphs = []; // Will store filtered paragraphs for current page
let currentPage = 1;
let pageSize = 25;
let totalPages = 1;
let totalItems = 0;
let currentFilters = {}; // Store current filter state
let compactModeEnabled = false;

// Cache for rendered paragraph elements
let visibleItemCache = new Map();
// Estimated average height of a paragraph card (will be dynamically updated)
let itemHeight = 100; // MODIFIED: Reduced from 220 to 100 to fix spacing issue
// Number of extra items to render above and below the visible area
let bufferSize = 5;

// Cache selectors for performance
let virtualScrollContainer;
let paragraphsListEl;
let loadingOverlay;
let batchActionBar;
let selectedCountEl;
let tagData = {};

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
  console.log("DOM loaded, initializing paragraph viewer");
  
  // Cache DOM elements
  virtualScrollContainer = document.getElementById('virtualScrollContainer');
  paragraphsListEl = document.getElementById('paragraphsList');
  loadingOverlay = document.getElementById('loadingOverlay');
  batchActionBar = document.getElementById('batchActionBar');
  selectedCountEl = document.getElementById('selectedCount');
  
  // Log element presence for debugging
  console.log('virtualScrollContainer found?', !!virtualScrollContainer);
  console.log('paragraphsListEl found?', !!paragraphsListEl);
  console.log('loadingOverlay found?', !!loadingOverlay);
  
  // Check for critical elements
  if (!virtualScrollContainer || !paragraphsListEl) {
    console.error("Critical DOM elements missing!");
    const errorMessage = document.createElement('div');
    errorMessage.className = 'alert alert-danger m-3';
    errorMessage.innerHTML = '<strong>Error:</strong> Required DOM elements not found. Please check the HTML structure.';
    document.body.insertBefore(errorMessage, document.body.firstChild);
    return;
  }
  
  // Extract tag data for easier access
  document.querySelectorAll('.tag-selection .tag-badge').forEach(tagBadge => {
    const tagId = tagBadge.getAttribute('data-tag-id');
    if (tagId) {
      tagData[tagId] = {
        name: tagBadge.textContent.trim(),
        color: tagBadge.style.backgroundColor
      };
    }
  });
  
  // Initialize UI components
  setupFilterEventListeners();
  setupPaginationControls();
  setupTagManagement();
  setupBatchSelection();
  setupCompactViewToggle();
  setupContentToggles();
  
  // Initialize empty state button
  const emptyStateResetBtn = document.getElementById('emptyStateResetBtn');
  if (emptyStateResetBtn) {
    emptyStateResetBtn.addEventListener('click', function() {
      resetFilters();
    });
  }
  
  // Initialize virtualized scrolling
  initVirtualScroll();
  
  // Check URL parameters for initial filter state
  initializeFiltersFromURL();
  
  // Load initial data
  console.log("Loading initial data...");
  loadParagraphsPage(1);
});

// Initialize virtualized scrolling
function initVirtualScroll() {
  // Create a spacer element to maintain scroll height
  const spacer = document.createElement('div');
  spacer.id = 'virtual-scroll-spacer';
  spacer.style.width = '100%';
  spacer.style.height = '0px';
  paragraphsListEl.appendChild(spacer);
  
  // Add scroll event listener (debounced for performance)
  if (virtualScrollContainer) {
    virtualScrollContainer.addEventListener('scroll', debounce(handleVirtualScroll, 10));
  } else {
    console.error("Virtual scroll container not found!");
  }
}

// Handle scroll events
function handleVirtualScroll() {
  updateVirtualScroll();
}

// Update virtual scroll rendering
function updateVirtualScroll() {
  if (!filteredParagraphs.length) {
    // Show empty state if no paragraphs
    const emptyState = document.getElementById('emptyState');
    if (emptyState) {
      emptyState.style.display = 'block';
    }
    return;
  } else {
    const emptyState = document.getElementById('emptyState');
    if (emptyState) {
      emptyState.style.display = 'none';
    }
  }
  
  if (!virtualScrollContainer) return;
  
  const scrollTop = virtualScrollContainer.scrollTop;
  const containerHeight = virtualScrollContainer.clientHeight;
  
  // Calculate which items should be visible
  const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - bufferSize);
  const endIndex = Math.min(
    filteredParagraphs.length, 
    Math.ceil((scrollTop + containerHeight) / itemHeight) + bufferSize
  );
  
  // Update spacer height to maintain scroll position
  const spacer = document.getElementById('virtual-scroll-spacer');
  if (spacer) {
    spacer.style.height = `${filteredParagraphs.length * itemHeight}px`;
  }
  
  // Get the current visible range
  const visibleRange = new Set();
  for (let i = startIndex; i < endIndex; i++) {
    visibleRange.add(i);
  }
  
  // Remove items that shouldn't be visible anymore
  visibleItemCache.forEach((element, index) => {
    if (!visibleRange.has(index)) {
      element.remove();
      visibleItemCache.delete(index);
    }
  });
  
  // Add items that should be visible
  for (let i = startIndex; i < endIndex; i++) {
    if (!visibleItemCache.has(i)) {
      const paraObj = filteredParagraphs[i];
      if (paraObj) {
        const element = createParagraphCard(paraObj);
        
        // Position the item absolutely
        element.style.position = 'absolute';
        element.style.top = `${i * itemHeight}px`;
        element.style.width = 'calc(100% - 16px)';
        element.style.left = '8px';
        
        paragraphsListEl.appendChild(element);
        visibleItemCache.set(i, element);
        
        // Set up event listeners for this card
        setupCardEventListeners(element);
      }
    }
  }
}

// Setup events for a newly created card
function setupCardEventListeners(card) {
  // Toggle paragraph content
  const readMoreBtn = card.querySelector('.read-more-btn');
  if (readMoreBtn) {
    readMoreBtn.addEventListener('click', function() {
      const preview = card.querySelector('.paragraph-preview');
      const fullContent = card.querySelector('.paragraph-full-content');
      
      if (fullContent.style.display === 'none') {
        preview.style.display = 'none';
        fullContent.style.display = 'block';
        
        // Update toggle button
        const toggleBtn = card.querySelector('.toggle-para-btn');
        if (toggleBtn) {
          toggleBtn.innerHTML = '<i class="bi bi-arrows-collapse"></i>';
          toggleBtn.title = 'Collapse';
        }
      }
    });
  }
  
  // Toggle button
  const toggleBtn = card.querySelector('.toggle-para-btn');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', function() {
      const preview = card.querySelector('.paragraph-preview');
      const fullContent = card.querySelector('.paragraph-full-content');
      
      if (fullContent.style.display === 'none') {
        preview.style.display = 'none';
        fullContent.style.display = 'block';
        this.innerHTML = '<i class="bi bi-arrows-collapse"></i>';
        this.title = 'Collapse';
      } else {
        preview.style.display = 'block';
        fullContent.style.display = 'none';
        this.innerHTML = '<i class="bi bi-arrows-expand"></i>';
        this.title = 'Expand';
      }
    });
  }
  
  // Stats button
  const statsBtn = card.querySelector('.show-stats-btn');
  if (statsBtn) {
    statsBtn.addEventListener('click', function() {
      const statsContainer = card.querySelector('.content-stats');
      if (statsContainer) {
        statsContainer.classList.toggle('d-none');
        
        // Update button icon
        if (statsContainer.classList.contains('d-none')) {
          this.innerHTML = '<i class="bi bi-info-circle"></i>';
        } else {
          this.innerHTML = '<i class="bi bi-x-circle"></i>';
        }
      }
    });
  }
  
  // Checkbox for batch selection
  const checkbox = card.querySelector('.paragraph-checkbox');
  if (checkbox) {
    checkbox.addEventListener('change', function() {
      const paraId = this.dataset.paraId;
      const paraObj = allParagraphs.find(p => p.id.toString() === paraId);
      
      if (paraObj) {
        paraObj.selected = this.checked;
        updateBatchSelectionUI();
      }
    });
  }
}

// Debounce function to prevent excessive calculations
function debounce(func, wait) {
  let timeout;
  return function(...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}

// Create a paragraph card element
function createParagraphCard(paraObj) {
  const card = document.createElement('div');
  card.className = `card paragraph-card ${paraObj.type} mb-2`; // MODIFIED: mb-3 to mb-2
  card.dataset.id = paraObj.id;
  card.dataset.type = paraObj.type;
  card.dataset.occurrences = paraObj.occurrences;
  card.dataset.document = paraObj.documentName;
  card.dataset.tags = paraObj.tags.map(t => t.id).join(',');
  card.dataset.contentLength = paraObj.contentLength;
  
  // Enhanced duplicate badge display
  const duplicateBadge = paraObj.appearsInMultiple ? 
    `<span class="badge bg-warning" title="This content appears in multiple documents">
      <i class="bi bi-files me-1"></i>Duplicate
    </span>` : '';
  
  // Make occurrence badge more informative
  const occurrenceBadge = 
    `<span class="badge bg-info occurrence-badge" onclick="toggleDocumentReferences(this, ${paraObj.id})" 
     title="${paraObj.occurrences > 1 ? 'Click to view all documents containing this paragraph' : 'Source document'}">
      <i class="bi bi-files me-1"></i>${paraObj.occurrences} ${paraObj.occurrences > 1 ? 'occurrences' : 'occurrence'}
    </span>`;
  
  // Generate HTML content based on paragraph object
  card.innerHTML = `
    <!-- Paragraph Selection Checkbox -->
    <div class="paragraph-select">
      <div class="form-check">
        <input class="form-check-input paragraph-checkbox" type="checkbox" id="selectPara${paraObj.id}" data-para-id="${paraObj.id}" ${paraObj.selected ? 'checked' : ''}>
        <label class="form-check-label" for="selectPara${paraObj.id}"></label>
      </div>
    </div>
    
    <div class="card-header py-1"> <!-- MODIFIED: Reduced padding from py-2 to py-1 -->
      <!-- All metadata, tags, and actions consolidated in the top row -->
      <div class="d-flex justify-content-between align-items-center">
        <!-- Type badges and occurrence info on the left -->
        <div class="d-flex align-items-center gap-2">
          <span class="badge ${getTypeColor(paraObj.type)}">${paraObj.type}</span>
          
          ${duplicateBadge}
          
          <!-- Occurrence badge (clickable) -->
          ${occurrenceBadge}
        </div>
        
        <!-- Tags and action buttons together on the right -->
        <div class="d-flex align-items-center">
          <!-- Tags moved to the right -->
          <div class="paragraph-tags d-flex flex-wrap gap-1 me-2">
            ${renderTags(paraObj.tags)}
          </div>
          
          <!-- Action buttons -->
          <div class="action-buttons d-flex align-items-center gap-1">
            <button class="btn btn-sm btn-light py-0 px-2" onclick="showTagModal(${paraObj.id})" title="Add Tag">
              <i class="bi bi-tag"></i>
            </button>
            <button class="btn btn-sm btn-light show-stats-btn py-0 px-2" title="Show Statistics">
              <i class="bi bi-info-circle"></i>
            </button>
            <a href="/paragraphs?document_id=${paraObj.documentId}" class="btn btn-sm btn-light py-0 px-2" title="View in Context">
              <i class="bi bi-eye"></i>
            </a>
            <button class="btn btn-sm btn-light toggle-para-btn ms-1" title="Expand/Collapse">
              <i class="bi bi-arrows-expand"></i>
            </button>
          </div>
        </div>
      </div>
    </div>
    
    <div class="card-body py-1"> <!-- MODIFIED: Reduced padding from py-2 to py-1 -->
      <!-- Document References - Shows where this paragraph appears -->
      <div class="document-references mb-1 p-2 rounded bg-light" id="docRefs-${paraObj.id}" style="display: none;"> <!-- MODIFIED: mb-2 to mb-1 -->
        <div class="d-flex justify-content-between align-items-center mb-1">
          <p class="text-muted mb-0 small">
            <i class="bi bi-files me-1"></i> 
            ${paraObj.occurrences > 1 ? 'Appears in multiple documents:' : 'Source document:'}
          </p>
        </div>
        <div class="d-flex flex-wrap gap-2">
          ${renderDocumentReferences(paraObj.documentReferences)}
        </div>
      </div>
      
      ${paraObj.headerContent ? 
        `<div class="mb-1 p-2 rounded bg-light"> <!-- MODIFIED: mb-2 to mb-1 -->
          <strong><i class="bi bi-type-h1 me-1"></i>Header:</strong> ${escapeHtml(paraObj.headerContent)}
         </div>` : ''}
      
      <!-- Content preview (always visible) -->
      <div class="paragraph-preview mb-1"> <!-- MODIFIED: mb-2 to mb-1 -->
        ${escapeHtml(truncateText(paraObj.content, 150))}
        <button class="btn btn-sm btn-link p-0 ms-1 read-more-btn">Read more</button>
      </div>
      
      <!-- Full content (hidden by default) -->
      <div class="paragraph-full-content p-2 bg-light rounded mb-1" style="display: none;"> <!-- MODIFIED: mb-2 to mb-1 -->
        ${paraObj.type === 'table' ? 
          `<pre class="table-content">${escapeHtml(paraObj.content)}</pre>` : 
          escapeHtml(paraObj.content)}
      </div>

      <!-- Content stats (hidden by default) -->
      <div class="content-stats p-2 bg-light rounded mb-1 d-none"> <!-- MODIFIED: mb-2 to mb-1 -->
        <div class="row text-center">
          <div class="col-md-3">
            <small class="text-muted">Characters</small>
            <div>${paraObj.contentLength}</div>
          </div>
          <div class="col-md-3">
            <small class="text-muted">Words</small>
            <div>${countWords(paraObj.content)}</div>
          </div>
          <div class="col-md-3">
            <small class="text-muted">Sentences</small>
            <div>${countSentences(paraObj.content)}</div>
          </div>
          <div class="col-md-3">
            <small class="text-muted">Position</small>
            <div>#${paraObj.position || 'N/A'}</div>
          </div>
        </div>
      </div>
    </div>
  `;
  
  return card;
}

/**
 * Load paragraphs for a specific page from the API
 * @param {number} page - The page number to load
 */
function loadParagraphsPage(page) {
  console.log("Loading paragraphs page:", page, "with filters:", currentFilters);
  
  // Hide empty state and show loading indicator if first page
  if (page === 1) {
    const emptyState = document.getElementById('emptyState');
    const initialLoadingIndicator = document.getElementById('initialLoadingIndicator');
    
    if (emptyState) emptyState.style.display = 'none';
    if (initialLoadingIndicator) initialLoadingIndicator.style.display = 'block';
  }
  
  showLoading('Loading paragraphs...');
  
  // Build the API URL with filters
  const url = new URL('/api/paragraphs', window.location.origin);
  
  // Add pagination parameters
  url.searchParams.append('page', page);
  url.searchParams.append('per_page', pageSize);
  
  // Add any active filters
  if (currentFilters.documentId) {
    url.searchParams.append('document_id', currentFilters.documentId);
  }
  
  if (currentFilters.paragraphType) {
    url.searchParams.append('type', currentFilters.paragraphType);
  }
  
  if (currentFilters.tagId) {
    url.searchParams.append('tag_id', currentFilters.tagId);
  }
  
  if (currentFilters.minLength) {
    url.searchParams.append('min_length', currentFilters.minLength);
  }
  
  if (currentFilters.searchQuery) {
    url.searchParams.append('search', currentFilters.searchQuery);
  }
  
  // FIXED: Always explicitly set the show_all_duplicates parameter with string values
  url.searchParams.append('show_all_duplicates', currentFilters.showAllDuplicates ? '1' : '0');
  
  // Add sorting parameters
  if (currentFilters.sortBy) {
    url.searchParams.append('sort_by', currentFilters.sortBy);
    url.searchParams.append('sort_direction', currentFilters.sortDirection || 'desc');
  }
  
  console.log("Fetching paragraphs from:", url.toString());
  
  // Fetch data from API
  fetch(url)
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      console.log("API returned data:", data);
      hideLoading();
      
      // Hide loading indicator
      const initialLoadingIndicator = document.getElementById('initialLoadingIndicator');
      if (initialLoadingIndicator) initialLoadingIndicator.style.display = 'none';
      
      // Update pagination information
      currentPage = data.current_page;
      totalPages = data.total_pages;
      totalItems = data.total_items;
      
      // Update pagination controls
      updatePaginationInfo();
      
      // Update duplicate indicator badge
      updateDuplicateIndicator();
      
      // Process received paragraphs
      if (data.paragraphs && data.paragraphs.length > 0) {
        console.log(`Received ${data.paragraphs.length} paragraphs`);
        
        // Clear existing data if it's page 1
        if (page === 1) {
          filteredParagraphs = [];
          allParagraphs = [];
          
          // Clear item cache when loading new data
          visibleItemCache.forEach((element) => {
            element.remove();
          });
          visibleItemCache.clear();
        }
        
        // Process paragraphs
        data.paragraphs.forEach(para => {
          const paraObj = {
            id: para.id,
            content: para.content,
            documentId: para.documentId,
            documentName: para.documentName,
            type: para.type,
            position: para.position,
            headerContent: para.headerContent,
            contentLength: para.contentLength,
            tags: para.tags || [],
            documentReferences: para.documentReferences || [],
            occurrences: para.occurrences || 1,
            appearsInMultiple: para.appearsInMultiple || false,
            selected: false
          };
          
          allParagraphs.push(paraObj);
          filteredParagraphs.push(paraObj);
        });
        
        // Update paragraph count display
        const paragraphCount = document.getElementById('paragraphCount');
        if (paragraphCount) paragraphCount.textContent = `${totalItems} paragraphs`;
        
        // Update virtual scroll to display the paragraphs
        updateVirtualScroll();
      } else {
        console.log("No paragraphs returned from API");
        
        // Show empty state if no paragraphs
        const emptyState = document.getElementById('emptyState');
        if (emptyState) emptyState.style.display = 'block';
        
        const paragraphCount = document.getElementById('paragraphCount');
        if (paragraphCount) paragraphCount.textContent = '0 paragraphs';
      }
    })
    .catch(error => {
      console.error("Error loading paragraphs:", error);
      hideLoading();
      
      const initialLoadingIndicator = document.getElementById('initialLoadingIndicator');
      if (initialLoadingIndicator) initialLoadingIndicator.style.display = 'none';
      
      // Show error message
      if (paragraphsListEl) {
        paragraphsListEl.innerHTML = `
          <div class="alert alert-danger mt-3">
            <h5>Error loading paragraphs</h5>
            <p>${error.message}</p>
            <button class="btn btn-primary" onclick="loadParagraphsPage(1)">Try Again</button>
          </div>
        `;
      }
      
      // Show error toast
      showToast('Failed to load paragraphs: ' + error.message, 'danger');
    });
}

/**
 * Update pagination information display
 */
function updatePaginationInfo() {
  // Update pagination counters
  const currentPageEls = document.querySelectorAll('#currentPage, #paginationCurrentPage');
  const totalPagesEls = document.querySelectorAll('#totalPages, #paginationTotalPages');
  
  currentPageEls.forEach(el => {
    if (el) el.textContent = currentPage;
  });
  
  totalPagesEls.forEach(el => {
    if (el) el.textContent = totalPages;
  });
  
  // Disable/enable pagination buttons as needed
  const prevButtons = document.querySelectorAll('#prevPageBtn, #firstPageBtn');
  const nextButtons = document.querySelectorAll('#nextPageBtn, #lastPageBtn');
  
  prevButtons.forEach(btn => {
    if (btn) btn.disabled = currentPage <= 1;
  });
  
  nextButtons.forEach(btn => {
    if (btn) btn.disabled = currentPage >= totalPages;
  });
}

/**
 * Update the duplicate indicator badge
 */
function updateDuplicateIndicator() {
  const indicatorBadge = document.getElementById('duplicateIndicator');
  if (indicatorBadge) {
    if (currentFilters.showAllDuplicates) {
      indicatorBadge.className = 'badge bg-info rounded-pill ms-2';
      indicatorBadge.innerHTML = '<i class="bi bi-files me-1"></i> Showing all duplicates';
    } else {
      indicatorBadge.className = 'badge bg-secondary rounded-pill ms-2';
      indicatorBadge.innerHTML = '<i class="bi bi-files me-1"></i> Duplicates collapsed';
    }
  }
}

/**
 * Initialize filters from URL parameters
 */
function initializeFiltersFromURL() {
  const urlParams = new URLSearchParams(window.location.search);
  
  // Extract filter values from URL
  const documentId = urlParams.get('document_id');
  
  // FIXED: Be explicit about parsing the show_all_duplicates parameter
  // Make sure we default to false (collapsed) when not specified
  const showAllDuplicatesParam = urlParams.get('show_all_duplicates');
  const showAllDuplicates = showAllDuplicatesParam === '1';
  
  // Set current filters with explicit default
  currentFilters = {
    documentId: documentId ? parseInt(documentId) : null,
    // FIXED: Always default to collapsed duplicates
    showAllDuplicates: showAllDuplicates || false,
    paragraphType: null,
    tagId: null,
    minLength: null,
    searchQuery: null,
    sortBy: 'occurrences',
    sortDirection: 'desc'
  };
  
  console.log("Initial filters:", currentFilters);
  
  // Update filter UI controls to match URL parameters
  if (documentId) {
    const documentSelect = document.getElementById('documentFilter');
    if (documentSelect) documentSelect.value = documentId;
  }
  
  // FIXED: Explicitly set the duplicate filter dropdown to match the current state
  const duplicateFilter = document.getElementById('duplicateFilter');
  if (duplicateFilter) {
    // Force the value to match our internal state
    duplicateFilter.value = currentFilters.showAllDuplicates ? 'all' : 'collapsed';
    console.log("Setting duplicate filter to:", duplicateFilter.value);
  }
  
  // Set default sort if not specified
  if (!currentFilters.sortBy) {
    currentFilters.sortBy = 'occurrences';
    currentFilters.sortDirection = 'desc';
    
    // Update the sort dropdown if it exists
    const sortOption = document.getElementById('sortOption');
    if (sortOption) sortOption.value = 'occurrences_desc';
  }
  
  // Update the duplicate indicator badge
  updateDuplicateIndicator();
}

/**
 * Set up event listeners for filter controls
 */
function setupFilterEventListeners() {
  // Document filter
  const documentFilter = document.getElementById('documentFilter');
  if (documentFilter) {
    documentFilter.addEventListener('change', function() {
      currentFilters.documentId = this.value ? parseInt(this.value) : null;
      loadParagraphsPage(1); // Reset to first page with new filter
    });
  }
  
  // Paragraph type filter
  const paragraphTypeFilter = document.getElementById('paragraphTypeFilter');
  if (paragraphTypeFilter) {
    paragraphTypeFilter.addEventListener('change', function() {
      currentFilters.paragraphType = this.value || null;
      loadParagraphsPage(1);
    });
  }
  
  // Tag filter
  const tagFilter = document.getElementById('tagFilter');
  if (tagFilter) {
    tagFilter.addEventListener('change', function() {
      currentFilters.tagId = this.value ? parseInt(this.value) : null;
      loadParagraphsPage(1);
    });
  }
  
  // Duplicate filter
  const duplicateFilter = document.getElementById('duplicateFilter');
  if (duplicateFilter) {
    duplicateFilter.addEventListener('change', function() {
      // Force Boolean interpretation with explicit comparison
      const showAll = this.value === 'all';
      console.log("Duplicate filter changed to:", this.value, "showAll =", showAll);
      
      // Update filter state
      currentFilters.showAllDuplicates = showAll;
      
      // For debugging - log the current state of filters
      console.log("Current filters after change:", JSON.stringify(currentFilters));
      
      // Update the duplicate indicator badge
      updateDuplicateIndicator();
      
      // Reset to page 1 and reload with new filter
      loadParagraphsPage(1);
    });
  }
  
  // Min length filter
  const minLengthFilter = document.getElementById('minLengthFilter');
  if (minLengthFilter) {
    minLengthFilter.addEventListener('input', debounce(function() {
      currentFilters.minLength = this.value ? parseInt(this.value) : null;
      loadParagraphsPage(1);
    }, 500));
  }
  
  // Search content
  const searchParagraphs = document.getElementById('searchParagraphs');
  if (searchParagraphs) {
    searchParagraphs.addEventListener('input', debounce(function() {
      currentFilters.searchQuery = this.value || null;
      loadParagraphsPage(1);
    }, 500));
  }
  
  // Clear search button
  const clearSearchBtn = document.getElementById('clearSearchBtn');
  if (clearSearchBtn) {
    clearSearchBtn.addEventListener('click', function() {
      if (searchParagraphs) {
        searchParagraphs.value = '';
        currentFilters.searchQuery = null;
        loadParagraphsPage(1);
      }
    });
  }
  
  // Sort option
  const sortOption = document.getElementById('sortOption');
  if (sortOption) {
    sortOption.addEventListener('change', function() {
      const value = this.value;
      if (value.includes('_')) {
        const [sortBy, direction] = value.split('_');
        currentFilters.sortBy = sortBy;
        currentFilters.sortDirection = direction;
      } else {
        currentFilters.sortBy = value;
        currentFilters.sortDirection = 'asc';
      }
      loadParagraphsPage(1);
    });
  }
}

/**
 * Reset all filters and reload paragraphs
 */
function resetFilters() {
  console.log("Resetting all filters to default values");
  
  // Reset filter values - default to collapsed duplicates
  currentFilters = {
    documentId: null,
    paragraphType: null,
    tagId: null,
    minLength: null,
    searchQuery: null,
    showAllDuplicates: false,  // Default to collapsed/show once
    sortBy: 'occurrences',
    sortDirection: 'desc'
  };
  
  // Reset UI elements
  const documentFilter = document.getElementById('documentFilter');
  if (documentFilter) documentFilter.value = '';
  
  const paragraphTypeFilter = document.getElementById('paragraphTypeFilter');
  if (paragraphTypeFilter) paragraphTypeFilter.value = '';
  
  const tagFilter = document.getElementById('tagFilter');
  if (tagFilter) tagFilter.value = '';
  
  const duplicateFilter = document.getElementById('duplicateFilter');
  if (duplicateFilter) duplicateFilter.value = 'collapsed';
  
  const minLengthFilter = document.getElementById('minLengthFilter');
  if (minLengthFilter) minLengthFilter.value = '0';
  
  const searchParagraphs = document.getElementById('searchParagraphs');
  if (searchParagraphs) searchParagraphs.value = '';
  
  const sortOption = document.getElementById('sortOption');
  if (sortOption) sortOption.value = 'occurrences_desc';
  
  // Update the duplicate indicator badge
  updateDuplicateIndicator();
  
  // Reload first page with reset filters
  loadParagraphsPage(1);
  
  // Show toast notification
  showToast('Filters reset to default values', 'info');
}

/**
 * Set up pagination control event listeners
 */
function setupPaginationControls() {
  const firstPageBtn = document.getElementById('firstPageBtn');
  const prevPageBtn = document.getElementById('prevPageBtn');
  const nextPageBtn = document.getElementById('nextPageBtn');
  const lastPageBtn = document.getElementById('lastPageBtn');
  const pageSizeSelect = document.getElementById('pageSizeSelect');
  
  if (firstPageBtn) {
    firstPageBtn.addEventListener('click', function() {
      if (currentPage > 1) {
        loadParagraphsPage(1);
      }
    });
  }
  
  if (prevPageBtn) {
    prevPageBtn.addEventListener('click', function() {
      if (currentPage > 1) {
        loadParagraphsPage(currentPage - 1);
      }
    });
  }
  
  if (nextPageBtn) {
    nextPageBtn.addEventListener('click', function() {
      if (currentPage < totalPages) {
        loadParagraphsPage(currentPage + 1);
      }
    });
  }
  
  if (lastPageBtn) {
    lastPageBtn.addEventListener('click', function() {
      if (currentPage < totalPages) {
        loadParagraphsPage(totalPages);
      }
    });
  }
  
  if (pageSizeSelect) {
    pageSizeSelect.addEventListener('change', function() {
      pageSize = parseInt(this.value);
      loadParagraphsPage(1);
    });
  }
}

/**
 * Set up batch selection functionality
 */
function setupBatchSelection() {
  // Select all visible button
  const selectAllBtn = document.getElementById('selectAllBtn');
  if (selectAllBtn) {
    selectAllBtn.addEventListener('click', selectAllVisible);
  }
  
  // Deselect all button
  const deselectAllBtn = document.getElementById('deselectAllBtn');
  if (deselectAllBtn) {
    deselectAllBtn.addEventListener('click', deselectAll);
  }
  
  // Cancel selection button
  const cancelSelectionBtn = document.getElementById('cancelSelectionBtn');
  if (cancelSelectionBtn) {
    cancelSelectionBtn.addEventListener('click', cancelSelection);
  }
  
  // Show/hide batch action bar based on selection
  updateBatchSelectionUI();
}

/**
 * Update batch selection UI based on selected paragraphs
 */
function updateBatchSelectionUI() {
  const selectedCount = allParagraphs.filter(p => p.selected).length;
  
  // Update selected count
  if (selectedCountEl) {
    selectedCountEl.textContent = `${selectedCount} selected`;
  }
  
  // Show/hide batch action bar
  if (batchActionBar) {
    if (selectedCount > 0) {
      batchActionBar.classList.add('visible');
    } else {
      batchActionBar.classList.remove('visible');
    }
  }
}

/**
 * Setup tag management functionality
 */
function setupTagManagement() {
  // Tag modal functionality is already handled by the showTagModal and addTag functions
}

/**
 * Set up compact view toggle
 */
function setupCompactViewToggle() {
  const viewToggleBtn = document.getElementById('viewToggleBtn');
  if (viewToggleBtn) {
    viewToggleBtn.addEventListener('click', function() {
      document.body.classList.toggle('compact-mode');
      compactModeEnabled = document.body.classList.contains('compact-mode');
      
      // Update the icon
      if (compactModeEnabled) {
        viewToggleBtn.innerHTML = '<i class="bi bi-layout-text-window"></i>';
        viewToggleBtn.title = 'Switch to Normal View';
        showToast('Compact view enabled');
      } else {
        viewToggleBtn.innerHTML = '<i class="bi bi-layout-text-window-reverse"></i>';
        viewToggleBtn.title = 'Switch to Compact View';
        showToast('Normal view enabled');
      }
      
      // Update the display
      updateVirtualScroll();
    });
  }
}

/**
 * Function to set up content toggles for read more/less
 */
function setupContentToggles() {
  // This is now handled individually for each card when it's created
  // The global buttons for expand/collapse all are handled here
  
  // Expand all paragraphs button
  const expandAllBtn = document.getElementById('expandAllParasBtn');
  if (expandAllBtn) {
    expandAllBtn.addEventListener('click', expandAllParagraphs);
  }
  
  // Collapse all paragraphs button
  const collapseAllBtn = document.getElementById('collapseAllParasBtn');
  if (collapseAllBtn) {
    collapseAllBtn.addEventListener('click', collapseAllParagraphs);
  }
}

/**
 * Expand all paragraphs that are currently visible
 */
function expandAllParagraphs() {
  showLoading('Expanding paragraphs...');
  
  visibleItemCache.forEach((card) => {
    const preview = card.querySelector('.paragraph-preview');
    const fullContent = card.querySelector('.paragraph-full-content');
    const toggleBtn = card.querySelector('.toggle-para-btn');
    
    if (preview && fullContent) {
      preview.style.display = 'none';
      fullContent.style.display = 'block';
      
      if (toggleBtn) {
        toggleBtn.innerHTML = '<i class="bi bi-arrows-collapse"></i>';
        toggleBtn.title = 'Collapse';
      }
    }
  });
  
  hideLoading();
  showToast('All visible paragraphs expanded');
}

/**
 * Collapse all paragraphs that are currently visible
 */
function collapseAllParagraphs() {
  showLoading('Collapsing paragraphs...');
  
  visibleItemCache.forEach((card) => {
    const preview = card.querySelector('.paragraph-preview');
    const fullContent = card.querySelector('.paragraph-full-content');
    const toggleBtn = card.querySelector('.toggle-para-btn');
    
    if (preview && fullContent) {
      preview.style.display = 'block';
      fullContent.style.display = 'none';
      
      if (toggleBtn) {
        toggleBtn.innerHTML = '<i class="bi bi-arrows-expand"></i>';
        toggleBtn.title = 'Expand';
      }
    }
  });
  
  hideLoading();
  showToast('All visible paragraphs collapsed');
}

/**
 * Function to toggle document references display
 * @param {HTMLElement} badge - The badge that was clicked
 * @param {number} paragraphId - The ID of the paragraph
 */
function toggleDocumentReferences(badge, paragraphId) {
  // Find the document references container
  const docRefsContainer = document.getElementById(`docRefs-${paragraphId}`);
  
  if (!docRefsContainer) return;
  
  // Toggle its display
  if (docRefsContainer.style.display === 'none') {
    // Hide all other document references first
    document.querySelectorAll('.document-references').forEach(container => {
      container.style.display = 'none';
    });
    
    // Remove active class from all badges
    document.querySelectorAll('.occurrence-badge').forEach(b => {
      b.classList.remove('active');
    });
    
    // Show this one
    docRefsContainer.style.display = 'block';
    badge.classList.add('active');
  } else {
    // Hide this one
    docRefsContainer.style.display = 'none';
    badge.classList.remove('active');
  }
}

/**
 * Select all visible paragraphs
 */
function selectAllVisible() {
  visibleItemCache.forEach((card) => {
    const checkbox = card.querySelector('.paragraph-checkbox');
    const paraId = checkbox ? checkbox.dataset.paraId : null;
    
    if (checkbox && paraId) {
      checkbox.checked = true;
      
      const paraObj = allParagraphs.find(p => p.id.toString() === paraId);
      if (paraObj) {
        paraObj.selected = true;
      }
    }
  });
  
  updateBatchSelectionUI();
  showToast('Selected all visible paragraphs');
}

/**
 * Deselect all paragraphs
 */
function deselectAll() {
  allParagraphs.forEach(para => {
    para.selected = false;
  });
  
  // Also uncheck any visible checkboxes
  document.querySelectorAll('.paragraph-checkbox').forEach(checkbox => {
    checkbox.checked = false;
  });
  
  updateBatchSelectionUI();
  showToast('Deselected all paragraphs');
}

/**
 * Cancel the current selection
 */
function cancelSelection() {
  deselectAll();
}

/**
 * Batch tag selected paragraphs
 * @param {number} tagId - The ID of the tag to apply
 */
function batchTagSelected(tagId) {
  const selectedParas = allParagraphs.filter(p => p.selected);
  
  if (selectedParas.length === 0) {
    showToast('No paragraphs selected', 'warning');
    return;
  }
  
  showLoading(`Tagging ${selectedParas.length} paragraphs...`);
  
  // Create promises for each tagging operation
  const tagPromises = selectedParas.map(para => {
    const formData = new FormData();
    formData.append('paragraph_id', para.id);
    formData.append('tag_id', tagId);
    formData.append('tag_all_duplicates', 'true');
    
    return fetch('/tag-paragraph', {
      method: 'POST',
      body: formData
    }).then(response => response.json());
  });
  
  // Wait for all tagging operations to complete
  Promise.all(tagPromises)
    .then(results => {
      hideLoading();
      
      const successCount = results.filter(r => r.success).length;
      
      if (successCount === selectedParas.length) {
        showToast(`Successfully tagged ${successCount} paragraphs`, 'success');
      } else {
        showToast(`Tagged ${successCount} out of ${selectedParas.length} paragraphs`, 'warning');
      }
      
      // Reload the current page to reflect changes
      loadParagraphsPage(currentPage);
      
      // Clear selection
      deselectAll();
    })
    .catch(error => {
      hideLoading();
      console.error('Error batch tagging paragraphs:', error);
      showToast('Error tagging paragraphs', 'danger');
    });
}

// Tag management functions
function showTagModal(paragraphId) {
  document.getElementById('paragraphIdForTag').value = paragraphId;
  new bootstrap.Modal(document.getElementById('tagModal')).show();
}

function addTag(tagId) {
  const paragraphId = document.getElementById('paragraphIdForTag').value;
  showLoading('Adding tag...');
  
  // Create FormData for more reliable parameter passing
  const formData = new FormData();
  formData.append('paragraph_id', paragraphId);
  formData.append('tag_id', tagId);
  formData.append('tag_all_duplicates', 'true');
  
  fetch('/tag-paragraph', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    hideLoading();
    
    if (data.success) {
      // Hide the modal
      const modal = bootstrap.Modal.getInstance(document.getElementById('tagModal'));
      modal.hide();
      
      showToast('Tag added to all instances successfully!');
      
      // Reload the current page to reflect changes
      loadParagraphsPage(currentPage);
    } else {
      showToast('Failed to add tag: ' + (data.message || 'Unknown error'), 'danger');
    }
  })
  .catch(error => {
    hideLoading();
    console.error('Error adding tag:', error);
    showToast('An error occurred while adding the tag', 'danger');
  });
}

function removeTag(paragraphId, tagId) {
  if (!confirm('Are you sure you want to remove this tag from all instances of this paragraph?')) {
    return;
  }
  
  showLoading('Removing tag...');
  
  // Create FormData for more reliable parameter passing
  const formData = new FormData();
  formData.append('paragraph_id', paragraphId);
  formData.append('tag_id', tagId);
  formData.append('untag_all_duplicates', 'true');
  
  fetch('/untag-paragraph', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    hideLoading();
    
    if (data.success) {
      showToast('Tag removed from all instances successfully!');
      
      // Reload the current page to reflect changes
      loadParagraphsPage(currentPage);
    } else {
      showToast('Failed to remove tag: ' + (data.message || 'Unknown error'), 'danger');
    }
  })
  .catch(error => {
    hideLoading();
    console.error('Error removing tag:', error);
    showToast('An error occurred while removing the tag', 'danger');
  });
}

/**
 * Helper function to get the appropriate badge color for paragraph type
 */
function getTypeColor(type) {
  switch(type) {
    case 'header':
      return 'bg-primary';
    case 'normal':
      return 'bg-secondary';
    case 'list':
      return 'bg-success';
    case 'table':
      return 'bg-danger';
    case 'boilerplate':
      return 'bg-warning';
    default:
      return 'bg-secondary';
  }
}

/**
 * Show or hide loading overlay
 */
function showLoading(message = 'Loading...') {
  if (loadingOverlay) {
    const loadingMessage = loadingOverlay.querySelector('#loadingMessage');
    if (loadingMessage) {
      loadingMessage.textContent = message;
    }
    loadingOverlay.style.display = 'flex';
  } else {
    console.error("Loading overlay not found!");
  }
}

function hideLoading() {
  if (loadingOverlay) {
    loadingOverlay.style.display = 'none';
  } else {
    console.error("Loading overlay not found!");
  }
}

/**
 * Show toast notification
 */
function showToast(message, type = 'success') {
  // Create toast container if it doesn't exist
  let toastContainer = document.querySelector('.toast-container');
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
    document.body.appendChild(toastContainer);
  }
  
  // Create toast element
  const toastEl = document.createElement('div');
  toastEl.className = `toast align-items-center text-white bg-${type} border-0`;
  toastEl.setAttribute('role', 'alert');
  toastEl.setAttribute('aria-live', 'assertive');
  toastEl.setAttribute('aria-atomic', 'true');
  
  // Toast content
  toastEl.innerHTML = `
    <div class="d-flex">
      <div class="toast-body">
        ${message}
      </div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
    </div>
  `;
  
  // Add to container
  toastContainer.appendChild(toastEl);
  
  // Initialize and show the toast
  const toast = new bootstrap.Toast(toastEl, {
    autohide: true,
    delay: 3000
  });
  toast.show();
  
  // Remove from DOM after it's hidden
  toastEl.addEventListener('hidden.bs.toast', function() {
    toastEl.remove();
  });
}

/**
 * Render tags for a paragraph
 */
function renderTags(tags) {
  if (!tags || tags.length === 0) return '';
  
  return tags.map(tag => `
    <span class="badge tag-badge" style="background-color: ${tag.color}">
      ${escapeHtml(tag.name)}
      <i class="bi bi-x ms-1 remove-tag-btn" onclick="removeTag(${tag.id})"></i>
    </span>
  `).join('');
}

/**
 * Render document references
 */
function renderDocumentReferences(docRefs) {
  if (!docRefs || docRefs.length === 0) return '';
  
  // For a single reference, render it simply
  if (docRefs.length === 1) {
    const doc = docRefs[0];
    return `
      <a href="/document/${doc.id}" class="text-decoration-none">
        <span class="badge bg-light text-dark border">
          <i class="bi bi-file-earmark-text me-1"></i>${escapeHtml(doc.filename)}
        </span>
      </a>
    `;
  }
  
  // For multiple references, add a heading
  return `
    <div class="mb-1 small text-muted">Appears in ${docRefs.length} documents:</div>
    <div class="d-flex flex-wrap gap-2">
      ${docRefs.map(doc => `
        <a href="/document/${doc.id}" class="text-decoration-none">
          <span class="badge bg-light text-dark border">
            <i class="bi bi-file-earmark-text me-1"></i>${escapeHtml(doc.filename)}
          </span>
        </a>
      `).join('')}
    </div>
  `;
}

/**
 * Count words in a string
 */
function countWords(str) {
  return str ? str.split(/\s+/).filter(word => word.length > 0).length : 0;
}

/**
 * Count sentences in a string (approximation)
 */
function countSentences(str) {
  if (!str) return 0;
  // Count periods, exclamation points, and question marks as sentence endings
  return (str.match(/[.!?]+/g) || []).length;
}

/**
 * Truncate text to specified length with ellipsis
 */
function truncateText(text, maxLength) {
  if (!text || text.length <= maxLength) return text || '';
  return text.substring(0, maxLength) + '...';
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(html) {
  if (!html) return '';
  const div = document.createElement('div');
  div.textContent = html;
  return div.innerHTML;
}

// Log that the spacing fix has been applied to help with debugging
console.log("SPACING FIX: Item height reduced to", itemHeight, "px");