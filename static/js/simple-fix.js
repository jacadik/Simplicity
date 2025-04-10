// Save this file as "simple-fix.js" and add to paragraphs.html 
// with a script tag: <script src="/static/js/simple-fix.js"></script>

document.addEventListener('DOMContentLoaded', function() {
  console.log("Simple Fix Script Loaded");
  
  // Create a status container at the top of the page
  const statusDiv = document.createElement('div');
  statusDiv.classList.add('container', 'mt-3');
  statusDiv.innerHTML = `
    <div class="card">
      <div class="card-header bg-primary text-white">
        <h5 class="mb-0">Paragraph Display Diagnostics</h5>
      </div>
      <div class="card-body">
        <div id="diagStatus">Checking system status...</div>
        <div class="mt-3">
          <button id="checkAPIBtn" class="btn btn-primary">Check API</button>
          <button id="loadParagraphsBtn" class="btn btn-success ms-2">Load Paragraphs</button>
          <button id="checkDBBtn" class="btn btn-warning ms-2">Check Database</button>
        </div>
      </div>
    </div>
  `;
  
  // Insert at the top of the content area
  const contentArea = document.querySelector('.compact-header') || document.body;
  document.body.insertBefore(statusDiv, contentArea);
  
  // Get the status display element
  const statusDisplay = document.getElementById('diagStatus');
  
  // Set up event handlers
  document.getElementById('checkAPIBtn').addEventListener('click', checkAPI);
  document.getElementById('loadParagraphsBtn').addEventListener('click', loadParagraphs);
  document.getElementById('checkDBBtn').addEventListener('click', checkDatabase);
  
  // Run initial check
  initialCheck();
  
  // Functions
  function initialCheck() {
    statusDisplay.innerHTML = `<div class="alert alert-info">Running initial system check...</div>`;
    
    // First check if we can reach the API
    fetch('/api/paragraphs?page=1&per_page=5')
      .then(response => {
        if (!response.ok) {
          throw new Error(`API returned status ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        if (!data.paragraphs || data.paragraphs.length === 0) {
          statusDisplay.innerHTML = `
            <div class="alert alert-warning">
              <h5>API is working but no paragraphs found</h5>
              <p>The API endpoint is responding correctly, but there are no paragraphs in the database.</p>
              <p><strong>Potential causes:</strong></p>
              <ul>
                <li>You haven't uploaded any documents yet</li>
                <li>Document processing failed</li>
                <li>Database issue preventing paragraph storage</li>
              </ul>
              <p><strong>Next steps:</strong></p>
              <ol>
                <li>Go to the home page and upload a document</li>
                <li>Check server logs for processing errors</li>
                <li>Click "Check Database" to verify database connectivity</li>
              </ol>
            </div>
          `;
        } else {
          statusDisplay.innerHTML = `
            <div class="alert alert-success">
              <h5>API working correctly!</h5>
              <p>Found ${data.total_items} paragraphs in the database.</p>
              <p>Click "Load Paragraphs" to display them directly.</p>
            </div>
          `;
          // Since API is working and has paragraphs, let's try loading them automatically
          loadParagraphs();
        }
      })
      .catch(error => {
        statusDisplay.innerHTML = `
          <div class="alert alert-danger">
            <h5>API Connection Error</h5>
            <p>${error.message}</p>
            <p><strong>Potential causes:</strong></p>
            <ul>
              <li>API route not defined correctly in app.py</li>
              <li>Server error when processing the request</li>
              <li>Database connection issue</li>
            </ul>
            <p><strong>Next steps:</strong></p>
            <ol>
              <li>Check Flask server logs for errors</li>
              <li>Verify the /api/paragraphs route in app.py</li>
              <li>Click "Check Database" to test database connectivity</li>
            </ol>
          </div>
        `;
      });
  }
  
  function checkAPI() {
    statusDisplay.innerHTML = `<div class="alert alert-info">Testing API connection...</div>`;
    
    fetch('/api/paragraphs?page=1&per_page=5')
      .then(response => {
        if (!response.ok) {
          throw new Error(`API returned status ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        statusDisplay.innerHTML = `
          <div class="alert alert-success">
            <h5>API Connection Successful</h5>
            <p>API returned data with ${data.total_items || 0} total paragraphs.</p>
            <pre class="mt-3 p-3 border bg-light"><code>${JSON.stringify(data, null, 2).substring(0, 500)}...</code></pre>
          </div>
        `;
      })
      .catch(error => {
        statusDisplay.innerHTML = `
          <div class="alert alert-danger">
            <h5>API Connection Failed</h5>
            <p>${error.message}</p>
          </div>
        `;
      });
  }
  
  function loadParagraphs() {
    statusDisplay.innerHTML = `<div class="alert alert-info">Loading paragraphs...</div>`;
    
    // Get the paragraphs container
    const paragraphsList = document.getElementById('paragraphsList');
    if (!paragraphsList) {
      statusDisplay.innerHTML = `
        <div class="alert alert-danger">
          <h5>Element Not Found</h5>
          <p>Could not find the paragraphsList element to display paragraphs.</p>
        </div>
      `;
      return;
    }
    
    // Clear existing content
    paragraphsList.innerHTML = '';
    
    // Fetch paragraphs
    fetch('/api/paragraphs?page=1&per_page=25')
      .then(response => {
        if (!response.ok) {
          throw new Error(`API returned status ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        if (!data.paragraphs || data.paragraphs.length === 0) {
          paragraphsList.innerHTML = `
            <div class="alert alert-warning m-3">
              <h5>No Paragraphs Found</h5>
              <p>There are no paragraphs in the database. Upload documents first.</p>
            </div>
          `;
          statusDisplay.innerHTML = `
            <div class="alert alert-warning">
              <h5>No Paragraphs to Display</h5>
              <p>API returned 0 paragraphs. Upload documents first.</p>
            </div>
          `;
          return;
        }
        
        // Update paragraph count
        const paragraphCount = document.getElementById('paragraphCount');
        if (paragraphCount) {
          paragraphCount.textContent = `${data.total_items} paragraphs`;
        }
        
        // Update pagination
        const currentPage = document.getElementById('currentPage');
        const totalPages = document.getElementById('totalPages');
        const paginationCurrentPage = document.getElementById('paginationCurrentPage');
        const paginationTotalPages = document.getElementById('paginationTotalPages');
        
        if (currentPage) currentPage.textContent = data.current_page;
        if (totalPages) totalPages.textContent = data.total_pages;
        if (paginationCurrentPage) paginationCurrentPage.textContent = data.current_page;
        if (paginationTotalPages) paginationTotalPages.textContent = data.total_pages;
        
        // Render paragraphs
        data.paragraphs.forEach(paragraph => {
          const card = document.createElement('div');
          card.className = 'card mb-3 paragraph-card';
          card.innerHTML = `
            <div class="card-header py-2 d-flex justify-content-between align-items-center">
              <div>
                <span class="badge bg-secondary">${paragraph.type || 'Unknown'}</span>
                <small class="ms-2 text-muted">From: <a href="/document/${paragraph.documentId}">${paragraph.documentName}</a></small>
              </div>
              <div>
                <button class="btn btn-sm py-0 px-1 expand-collapse-btn">
                  <i class="bi bi-chevron-down"></i>
                </button>
              </div>
            </div>
            <div class="card-body py-2">
              <div class="paragraph-preview">${paragraph.content.substring(0, 150)}${paragraph.content.length > 150 ? '...' : ''}</div>
              <div class="paragraph-full-content mt-2" style="display: none;">${paragraph.content}</div>
            </div>
          `;
          
          // Add expand/collapse functionality
          card.querySelector('.expand-collapse-btn').addEventListener('click', function() {
            const preview = card.querySelector('.paragraph-preview');
            const full = card.querySelector('.paragraph-full-content');
            const icon = this.querySelector('i');
            
            if (full.style.display === 'none') {
              preview.style.display = 'none';
              full.style.display = 'block';
              icon.classList.replace('bi-chevron-down', 'bi-chevron-up');
            } else {
              preview.style.display = 'block';
              full.style.display = 'none';
              icon.classList.replace('bi-chevron-up', 'bi-chevron-down');
            }
          });
          
          paragraphsList.appendChild(card);
        });
        
        statusDisplay.innerHTML = `
          <div class="alert alert-success">
            <h5>Paragraphs Loaded Successfully</h5>
            <p>Displayed ${data.paragraphs.length} of ${data.total_items} total paragraphs.</p>
          </div>
        `;
      })
      .catch(error => {
        statusDisplay.innerHTML = `
          <div class="alert alert-danger">
            <h5>Failed to Load Paragraphs</h5>
            <p>${error.message}</p>
          </div>
        `;
      });
  }
  
  function checkDatabase() {
    statusDisplay.innerHTML = `<div class="alert alert-info">Checking database status...</div>`;
    
    // Create a simple endpoint check that will test database connectivity
    fetch('/api/paragraph-statistics')
      .then(response => {
        if (!response.ok) {
          throw new Error(`API returned status ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        if (data.error) {
          throw new Error(data.message || 'Database error');
        }
        
        statusDisplay.innerHTML = `
          <div class="alert alert-success">
            <h5>Database Connection Successful</h5>
            <p>Total paragraphs in database: ${data.total_paragraphs}</p>
            <p>Unique paragraphs: ${data.unique_paragraphs}</p>
            <p>Duplicate paragraphs: ${data.duplicate_paragraphs}</p>
            <div class="mt-2">
              <strong>Paragraph types:</strong>
              <ul>
                ${data.by_type.map(type => `<li>${type.type}: ${type.count}</li>`).join('')}
              </ul>
            </div>
          </div>
        `;
      })
      .catch(error => {
        statusDisplay.innerHTML = `
          <div class="alert alert-danger">
            <h5>Database Connection Error</h5>
            <p>${error.message}</p>
            <p><strong>Potential causes:</strong></p>
            <ul>
              <li>Database server not running</li>
              <li>Incorrect database credentials</li>
              <li>Database schema issues</li>
            </ul>
            <p><strong>Check your database connection in app.py:</strong></p>
            <pre class="p-2 bg-light border">DB_URL = "postgresql://paragraph_user:pass@localhost/paragraph_analyzer"</pre>
          </div>
        `;
      });
  }
});
