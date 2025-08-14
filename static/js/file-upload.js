// File upload handling JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize file inputs
    setupFileInput('pdfFile', 'pdfDisplay');
    setupFileInput('xmlFile', 'xmlDisplay');
});

function setupFileInput(inputId, displayId) {
    const input = document.getElementById(inputId);
    const display = document.getElementById(displayId);
    
    if (!input || !display) return;
    
    input.addEventListener('change', function(e) {
        const file = e.target.files[0];
        const textSpan = display.querySelector('.file-input-text');
        
        if (file) {
            textSpan.textContent = file.name;
            textSpan.classList.add('has-file');
            display.classList.add('has-file');
        } else {
            const defaultText = inputId === 'pdfFile' ? 
                'Choose PDF file or drag & drop' : 
                'Choose XML file (optional)';
            textSpan.textContent = defaultText;
            textSpan.classList.remove('has-file');
            display.classList.remove('has-file');
        }
    });

    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        display.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        display.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        display.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        display.style.borderColor = '#667eea';
        display.style.backgroundColor = '#f0f4ff';
    }

    function unhighlight() {
        if (!display.classList.contains('has-file')) {
            display.style.borderColor = '#ddd';
            display.style.backgroundColor = '#fafafa';
        }
    }

    display.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            const file = files[0];
            const expectedType = inputId === 'pdfFile' ? 'pdf' : 'xml';
            
            if (file.name.toLowerCase().endsWith(`.${expectedType}`)) {
                input.files = files;
                const event = new Event('change', { bubbles: true });
                input.dispatchEvent(event);
            } else {
                showResult(`Please select a valid ${expectedType.toUpperCase()} file.`, 'error');
            }
        }
    }
}
