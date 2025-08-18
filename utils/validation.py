class ValidationError(Exception):
    pass

def validate_form_data(request):
    # --- Validate PDF file ---
    if 'pdf_file' not in request.files:
        raise ValidationError('No PDF file provided')

    pdf_file = request.files['pdf_file']
    if pdf_file.filename == '':
        raise ValidationError('No PDF file selected')

    if not allowed_file(pdf_file.filename, 'pdf'):
        raise ValidationError('Invalid PDF file format')

    # --- Validate global page number input (for PDF) ---
    page_input = request.form.get('page_num', '').strip()
    if not page_input:
        raise ValidationError('Page number is required')

    def parse_page_numbers(page_input: str):
        pages = []
        for part in page_input.split(","):
            part = part.strip()
            if "-" in part:  # range like 8-10
                try:
                    start, end = map(int, part.split("-"))
                    if start < 1 or end < 1:
                        raise ValidationError("Page numbers must be positive")
                    if start > end:
                        raise ValidationError(f"Invalid page range: {part}")
                    pages.extend(range(start, end + 1))
                except ValueError:
                    raise ValidationError(f"Invalid page range format: {part}")
            else:  # single page
                try:
                    num = int(part)
                    if num < 1:
                        raise ValidationError("Page numbers must be positive")
                    pages.append(num)
                except ValueError:
                    raise ValidationError(f"Invalid page number: {part}")
        return sorted(set(pages))  # unique & sorted

    page_list = parse_page_numbers(page_input)
    if not page_list:
        raise ValidationError("No valid page numbers provided")

    # --- Validate XML uploads (multiple allowed) ---
    xml_files = []
    idx = 0
    while f"xml_files[{idx}]" in request.files:
        xml_file = request.files[f"xml_files[{idx}]"]
        xml_page_input = request.form.get(f"xml_pages[{idx}]", "").strip()

        if xml_file and xml_file.filename:
            if not allowed_file(xml_file.filename, 'xml'):
                raise ValidationError(f"Invalid XML file format: {xml_file.filename}")
            if not xml_page_input:
                raise ValidationError(f"Page number missing for XML file {xml_file.filename}")

            xml_page_list = parse_page_numbers(xml_page_input)
            if not xml_page_list:
                raise ValidationError(f"No valid pages for XML file {xml_file.filename}")

            xml_files.append({
                "file": xml_file,
                "filename": xml_file.filename,
                "pages": xml_page_list
            })
        idx += 1

    # --- Settings ---
    enable_network_colors = request.form.get('enable_network_colors') == 'true'

    return {
        'page_num': page_input,    # e.g. "8-10, 12"
        'page_list': page_list,    # e.g. [8,9,10,12]
        'enable_network_colors': enable_network_colors,
        'xml_files': xml_files     # list of {file, filename, pages}
    }


def allowed_file(filename, extension):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() == extension.lower()
