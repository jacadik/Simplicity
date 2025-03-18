function updateThresholdValue(value) {
    const thresholdValueElement = document.getElementById('thresholdValue');
    if (thresholdValueElement) {
        thresholdValueElement.textContent = value;
    }
}

document.addEventListener('DOMContentLoaded', function() {
    // Add search functionality for clusters
    const searchInput = document.getElementById('clusterSearch');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            const rows = document.querySelectorAll('.cluster-row');
            
            rows.forEach(row => {
                const name = row.cells[1].textContent.toLowerCase();
                const id = row.cells[0].textContent.toLowerCase();
                if (name.includes(searchTerm) || id.includes(searchTerm)) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        });
    }
    
    // Add explanation tooltips
    const similarityTypeSelect = document.getElementById('similarityTypeSelect');
    
    if (similarityTypeSelect) {
        similarityTypeSelect.addEventListener('change', function() {
            const selectedValue = this.value;
            const thresholdSlider = document.getElementById('thresholdSlider');
            
            if (thresholdSlider) {
                if (selectedValue === 'content') {
                    // Content similarity typically has lower thresholds
                    thresholdSlider.value = 0.8;
                } else {
                    // Text similarity may need higher thresholds
                    thresholdSlider.value = 0.85;
                }
                
                // Update the displayed value
                updateThresholdValue(thresholdSlider.value);
            }
        });
    }
    
    // Initialize threshold display
    const thresholdSlider = document.getElementById('thresholdSlider');
    if (thresholdSlider) {
        updateThresholdValue(thresholdSlider.value);
    }
    
    // Add hover effect to cluster rows
    const clusterRows = document.querySelectorAll('.cluster-row');
    clusterRows.forEach(row => {
        row.addEventListener('mouseenter', function() {
            this.classList.add('bg-light');
        });
        row.addEventListener('mouseleave', function() {
            this.classList.remove('bg-light');
        });
    });
});
