document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('analysisForm');
    
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }
    
    // Auto-hide result when form changes
    form.addEventListener('input', function() {
        const resultContainer = document.getElementById('resultContainer');
        if (resultContainer) {
            resultContainer.style.display = 'none';
        }
    });
});

async function handleFormSubmit(e) {
    e.preventDefault();
    
    const submitBtn = document.getElementById('submitBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const btnText = document.getElementById('btnText');
    
    // Show loading state
    submitBtn.disabled = true;
    loadingSpinner.style.display = 'inline-block';
    btnText.textContent = 'Processing...';
    
    // Prepare form data
    const formData = new FormData();
    const pdfFile = document.getElementById('pdfFile').files[0];
    const xmlFile = document.getElementById('xmlFile').files[0];
    
    formData.append('pdf_file', pdfFile);
    formData.append('page_num', document.getElementById('pageNum').value);
    formData.append('image_path', document.getElementById('imagePath').value);
    formData.append('enable_network_colors', document.getElementById('networkColors').checked);
    
    if (xmlFile) {
        formData.append('xml_file', xmlFile);
    }
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok && result.success) {
            showResults(result.result);
            showResult('Analysis completed successfully! Check the results below.', 'success');
        } else {
            showResult(result.error || 'An error occurred during analysis.', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showResult('Network error occurred. Please check your connection and try again.', 'error');
    } finally {
        // Reset button state
        submitBtn.disabled = false;
        loadingSpinner.style.display = 'none';
        btnText.textContent = 'Start Analysis';
    }
}

function showResult(message, type) {
    const container = document.getElementById('resultContainer');
    const content = document.getElementById('resultContent');
    
    if (container && content) {
        container.className = `result-container ${type}`;
        content.textContent = message;
        container.style.display = 'block';
        
        // Scroll to result
        container.scrollIntoView({ behavior: 'smooth' });
    }
}

function showResults(data) {
    // Create results display container
    const existingResults = document.getElementById('analysisResults');
    if (existingResults) {
        existingResults.remove();
    }
    
    // Create new results container
    const resultsContainer = document.createElement('div');
    resultsContainer.id = 'analysisResults';
    resultsContainer.innerHTML = createResultsHTML();
    
    // Insert after the form
    const form = document.getElementById('analysisForm');
    form.parentNode.insertBefore(resultsContainer, form.nextSibling);
    
    // Populate with actual data
    populateResultsData(data);
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

function createResultsHTML() {
    return `
        <div class="results-container">
            <div class="results-header">
                <h2>🔌 Analysis Results</h2>
                <p>Comprehensive analysis of your electrical circuit diagrams</p>
            </div>

            <div class="results-tabs">
                <button class="results-tab active" onclick="showResultsTab('dataframes-results')">📊 Data Tables</button>
                <button class="results-tab" onclick="showResultsTab('analysis-images-results')">🖼️ Analysis Images</button>
                <button class="results-tab" onclick="showResultsTab('drawn-lines-results')">📝 Drawn Lines</button>
            </div>

            <!-- Data Tables Tab -->
            <div id="dataframes-results" class="results-tab-content active">
                <div class="results-table-container">
                    <div class="results-table-header">
                        <div class="results-table-title">Wire Analysis Data</div>
                        <button class="results-download-btn" onclick="downloadResultsCSV('wire-data')">
                            📥 Download CSV
                        </button>
                    </div>
                    <div class="results-table-wrapper">
                        <table id="results-wire-table">
                            <thead id="results-wire-thead"></thead>
                            <tbody id="results-wire-tbody"></tbody>
                        </table>
                    </div>
                </div>

                <div class="results-table-container">
                    <div class="results-table-header">
                        <div class="results-table-title">Connection Analysis Data</div>
                        <button class="results-download-btn" onclick="downloadResultsCSV('connection-data')">
                            📥 Download CSV
                        </button>
                    </div>
                    <div class="results-table-wrapper">
                        <table id="results-connection-table">
                            <thead id="results-connection-thead"></thead>
                            <tbody id="results-connection-tbody"></tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- Analysis Images Tab -->
            <div id="analysis-images-results" class="results-tab-content">
                <div class="results-image-container">
                    <div class="results-image-header">🖼️ Combined Canvas</div>
                    <div class="results-image-display">
                        <img id="results-combined-canvas" src="" alt="Combined Canvas" />
                    </div>
                </div>

                <div class="results-image-container">
                    <div class="results-image-header">📍 Junction Points</div>
                    <div class="results-image-display">
                        <img id="results-junction-points" src="" alt="Junction Points" />
                    </div>
                </div>

                <div class="results-image-container">
                    <div class="results-image-header">📏 Line Canvas</div>
                    <div class="results-image-display">
                        <img id="results-line-canvas" src="" alt="Line Canvas" />
                    </div>
                </div>
            </div>

            <!-- Drawn Lines Tab -->
            <div id="drawn-lines-results" class="results-tab-content">
                <div class="results-pagination-container">
                    <div class="results-pagination-header">
                        <div class="results-pagination-title">Drawn Lines Collection</div>
                        <div class="results-pagination-info">
                            Showing <span id="results-current-range">0-0</span> of <span id="results-total-images">0</span> images
                        </div>
                    </div>

                    <div class="results-pagination-controls">
                        <button class="results-page-btn" id="results-prev-btn" onclick="changeResultsPage(-1)">← Previous</button>
                        <span class="results-current-page">Page <span id="results-current-page">1</span> of <span id="results-total-pages">1</span></span>
                        <button class="results-page-btn" id="results-next-btn" onclick="changeResultsPage(1)">Next →</button>
                    </div>

                    <div class="results-images-grid" id="results-images-grid">
                        <!-- Images will be dynamically populated here -->
                    </div>
                </div>
            </div>
        </div>
        
        <style>
            .results-container {
                margin: 30px 0;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                overflow: hidden;
                backdrop-filter: blur(10px);
            }

            .results-header {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }

            .results-header h2 {
                font-size: 2.2rem;
                margin-bottom: 10px;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }

            .results-tabs {
                display: flex;
                background: #f8f9fa;
                border-bottom: 2px solid #e9ecef;
            }

            .results-tab {
                flex: 1;
                padding: 15px 20px;
                background: none;
                border: none;
                cursor: pointer;
                font-size: 1.1rem;
                font-weight: 600;
                color: #6c757d;
                transition: all 0.3s ease;
                position: relative;
            }

            .results-tab:hover {
                background: rgba(79, 172, 254, 0.1);
                color: #4facfe;
            }

            .results-tab.active {
                color: #4facfe;
                background: white;
            }

            .results-tab.active::after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, #4facfe, #00f2fe);
            }

            .results-tab-content {
                display: none;
                padding: 30px;
            }

            .results-tab-content.active {
                display: block;
            }

            .results-table-container {
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                overflow: hidden;
                margin-bottom: 20px;
            }

            .results-table-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }

            .results-table-title {
                font-size: 1.5rem;
                font-weight: 600;
            }

            .results-download-btn {
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
                color: white;
                padding: 10px 20px;
                border-radius: 25px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s ease;
                backdrop-filter: blur(10px);
            }

            .results-download-btn:hover {
                background: rgba(255, 255, 255, 0.3);
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            }

            .results-table-wrapper {
                overflow-x: auto;
                max-height: 400px;
                overflow-y: auto;
            }

            .results-table-wrapper table {
                width: 100%;
                border-collapse: collapse;
            }

            .results-table-wrapper th, .results-table-wrapper td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #e9ecef;
            }

            .results-table-wrapper th {
                background: #f8f9fa;
                font-weight: 600;
                color: #495057;
                position: sticky;
                top: 0;
                z-index: 10;
            }

            .results-table-wrapper tr:hover {
                background: rgba(79, 172, 254, 0.05);
            }

            .results-image-container {
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                overflow: hidden;
                margin-bottom: 30px;
            }

            .results-image-header {
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-size: 1.3rem;
                font-weight: 600;
            }

            .results-image-display {
                padding: 20px;
                text-align: center;
            }

            .results-image-display img {
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease;
                cursor: pointer;
            }

            .results-image-display img:hover {
                transform: scale(1.02);
            }

            .results-pagination-container {
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-bottom: 20px;
            }

            .results-pagination-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }

            .results-pagination-title {
                font-size: 1.3rem;
                font-weight: 600;
                color: #495057;
            }

            .results-pagination-info {
                color: #6c757d;
                font-size: 0.9rem;
            }

            .results-pagination-controls {
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 10px;
                margin: 20px 0;
            }

            .results-page-btn {
                padding: 8px 16px;
                border: 1px solid #dee2e6;
                background: white;
                color: #495057;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s ease;
                font-weight: 500;
            }

            .results-page-btn:hover:not(:disabled) {
                background: #4facfe;
                color: white;
                border-color: #4facfe;
                transform: translateY(-2px);
            }

            .results-page-btn.active {
                background: #4facfe;
                color: white;
                border-color: #4facfe;
            }

            .results-page-btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }

            .results-current-page {
                font-weight: 600;
                color: #4facfe;
            }

            .results-images-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                padding: 20px;
            }

            .results-grid-image {
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                overflow: hidden;
                transition: transform 0.3s ease;
                cursor: pointer;
            }

            .results-grid-image:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            }

            .results-grid-image img {
                width: 100%;
                height: 200px;
                object-fit: cover;
            }

            .results-grid-image-info {
                padding: 15px;
                text-align: center;
                font-weight: 600;
                color: #495057;
            }

            @media (max-width: 768px) {
                .results-header h2 {
                    font-size: 1.8rem;
                }

                .results-tab-content {
                    padding: 20px;
                }

                .results-table-header {
                    flex-direction: column;
                    gap: 15px;
                    text-align: center;
                }

                .results-pagination-header {
                    flex-direction: column;
                    gap: 10px;
                    text-align: center;
                }

                .results-images-grid {
                    grid-template-columns: 1fr;
                    padding: 10px;
                }
            }
        </style>
    `;
}

// Global variables for results pagination
let resultsCurrentPage = 1;
let resultsImagesPerPage = 6;
let resultsTotalImages = 0;
let resultsData = null;

function populateResultsData(data) {
    resultsData = data;
    
    // Populate tables
    if (data.wire_data && data.wire_data.length > 0) {
        populateResultsTable('wire', data.wire_data);
    }
    
    if (data.connection_data && data.connection_data.length > 0) {
        populateResultsTable('connection', data.connection_data);
    }
    
    // Load images
    if (data.images) {
        const combinedCanvas = document.getElementById('results-combined-canvas');
        const junctionPoints = document.getElementById('results-junction-points');
        const lineCanvas = document.getElementById('results-line-canvas');
        
        if (data.images.combined_canvas && combinedCanvas) {
            combinedCanvas.src = data.images.combined_canvas;
        }
        
        if (data.images.junction_points && junctionPoints) {
            junctionPoints.src = data.images.junction_points;
        }
        
        if (data.images.line_canvas && lineCanvas) {
            lineCanvas.src = data.images.line_canvas;
        }
    }
    
    // Initialize pagination for drawn lines
    if (data.drawn_lines && data.drawn_lines.length > 0) {
        initializeResultsPagination(data.drawn_lines);
    }
}

function populateResultsTable(tableType, data) {
    const thead = document.getElementById(`results-${tableType}-thead`);
    const tbody = document.getElementById(`results-${tableType}-tbody`);
    
    if (!thead || !tbody || data.length === 0) return;
    
    // Clear existing content
    thead.innerHTML = '';
    tbody.innerHTML = '';
    
    // Create header
    const headerRow = document.createElement('tr');
    Object.keys(data[0]).forEach(key => {
        const th = document.createElement('th');
        th.textContent = key.replace(/_/g, ' ').toUpperCase();
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    
    // Create rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        Object.values(row).forEach(value => {
            const td = document.createElement('td');
            td.textContent = value !== null && value !== undefined ? value : 'N/A';
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
}

function initializeResultsPagination(images) {
    resultsTotalImages = images.length;
    resultsCurrentPage = 1;
    
    const totalPages = Math.ceil(resultsTotalImages / resultsImagesPerPage);
    
    document.getElementById('results-total-images').textContent = resultsTotalImages;
    document.getElementById('results-total-pages').textContent = totalPages;
    
    displayResultsImages(resultsCurrentPage, images);
    updateResultsPaginationControls();
}

function displayResultsImages(page, images) {
    const grid = document.getElementById('results-images-grid');
    if (!grid) return;
    
    grid.innerHTML = '';
    
    const startIndex = (page - 1) * resultsImagesPerPage;
    const endIndex = Math.min(startIndex + resultsImagesPerPage, images.length);
    
    document.getElementById('results-current-range').textContent = `${startIndex + 1}-${endIndex}`;
    document.getElementById('results-current-page').textContent = page;
    
    for (let i = startIndex; i < endIndex; i++) {
        const imageDiv = document.createElement('div');
        imageDiv.className = 'results-grid-image';
        
        imageDiv.innerHTML = `
            <img src="${images[i].image}" alt="${images[i].name}" onclick="openImageModal('${images[i].image}', '${images[i].name}')" />
            <div class="results-grid-image-info">${images[i].name}</div>
        `;
        
        grid.appendChild(imageDiv);
    }
}

function changeResultsPage(direction) {
    if (!resultsData || !resultsData.drawn_lines) return;
    
    const totalPages = Math.ceil(resultsTotalImages / resultsImagesPerPage);
    const newPage = resultsCurrentPage + direction;
    
    if (newPage >= 1 && newPage <= totalPages) {
        resultsCurrentPage = newPage;
        displayResultsImages(resultsCurrentPage, resultsData.drawn_lines);
        updateResultsPaginationControls();
    }
}

function updateResultsPaginationControls() {
    const totalPages = Math.ceil(resultsTotalImages / resultsImagesPerPage);
    
    const prevBtn = document.getElementById('results-prev-btn');
    const nextBtn = document.getElementById('results-next-btn');
    
    if (prevBtn) prevBtn.disabled = resultsCurrentPage === 1;
    if (nextBtn) nextBtn.disabled = resultsCurrentPage === totalPages;
}

function showResultsTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.results-tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Remove active class from all tabs
    document.querySelectorAll('.results-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab content
    const targetTab = document.getElementById(tabName);
    if (targetTab) {
        targetTab.classList.add('active');
    }
    
    // Add active class to clicked tab
    event.target.classList.add('active');
}

function downloadResultsCSV(dataType) {
    if (!resultsData) return;
    
    let data, filename;
    
    if (dataType === 'wire-data') {
        data = resultsData.wire_data;
        filename = 'wire_analysis.csv';
    } else if (dataType === 'connection-data') {
        data = resultsData.connection_data;
        filename = 'connection_analysis.csv';
    }
    
    if (!data || data.length === 0) {
        alert('No data available to download.');
        return;
    }
    
    const headers = Object.keys(data[0]);
    const csvContent = [
        headers.join(','),
        ...data.map(row => headers.map(header => {
            const value = row[header];
            // Escape commas and quotes in CSV values
            const escaped = String(value || '').replace(/"/g, '""');
            return escaped.includes(',') || escaped.includes('"') || escaped.includes('\n') 
                ? `"${escaped}"` 
                : escaped;
        }).join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

function openImageModal(imageSrc, imageName) {
    // Create modal for full-size image viewing
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.9);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 10000;
        cursor: pointer;
    `;
    
    const img = document.createElement('img');
    img.src = imageSrc;
    img.alt = imageName;
    img.style.cssText = `
        max-width: 90%;
        max-height: 90%;
        object-fit: contain;
        border-radius: 8px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    `;
    
    const title = document.createElement('div');
    title.textContent = imageName;
    title.style.cssText = `
        position: absolute;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        background: rgba(0, 0, 0, 0.7);
        padding: 10px 20px;
        border-radius: 25px;
        backdrop-filter: blur(10px);
    `;
    
    modal.appendChild(img);
    modal.appendChild(title);
    
    modal.addEventListener('click', () => {
        document.body.removeChild(modal);
    });
    
    document.body.appendChild(modal);
}