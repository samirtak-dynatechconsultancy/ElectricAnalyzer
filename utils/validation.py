class ValidationError(Exception):
    pass

def validate_form_data(request):
    # Validate PDF file
    if 'pdf_file' not in request.files:
        raise ValidationError('No PDF file provided')
    
    pdf_file = request.files['pdf_file']
    if pdf_file.filename == '':
        raise ValidationError('No PDF file selected')
    
    if not allowed_file(pdf_file.filename, 'pdf'):
        raise ValidationError('Invalid PDF file format')
    
    # Validate page number
    try:
        page_num = int(request.form.get('page_num', 1))
        if page_num < 1:
            raise ValidationError('Page number must be positive')
    except (ValueError, TypeError):
        raise ValidationError('Invalid page number')
    
    
    # Get network colors setting
    enable_network_colors = request.form.get('enable_network_colors') == 'true'
    
    return {
        'page_num': page_num,
        'enable_network_colors': enable_network_colors
    }

def allowed_file(filename, extension):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() == extension.lower()
