import subprocess
import os

def optimize_pdf(input_pdf, output_pdf, quality="screen"):
    """
    Optimize and clean a PDF file using Ghostscript.
    
    Parameters:
        input_pdf (str): Path to the input PDF file
        output_pdf (str): Path where the optimized PDF will be saved
        quality (str): One of ['screen', 'ebook', 'printer', 'prepress', 'default']
                       - screen: smallest, low quality
                       - ebook: better quality, small size
                       - printer: good for printing
                       - prepress: high quality, larger file
                       - default: standard optimization
    """
    gs_path = r"C:\Program Files\gs\gs10.06.0\bin\gswin64c.exe"
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    # Ghostscript command
    command = [
        "gswin64c.exe",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/" + quality,
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        f"-sOutputFile={output_pdf}",
        input_pdf
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ PDF optimized and saved to: {output_pdf}")
    except subprocess.CalledProcessError as e:
        print("❌ Ghostscript failed:", e)

# # Example usage:
# input_path = r"C:\Users\Samir.Tak\Downloads\OneDrive_1_1-9-2025\743)CEP-ATIT1.pdf"
# output_path = r"C:\Users\Samir.Tak\Downloads\OneDrive_1_1-9-2025\743)CEP-ATIT1_optimized_2.pdf"
# optimize_pdf(input_path, output_path, quality="ebook")
