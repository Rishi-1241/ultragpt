import os
from pathlib import Path
from tempfile import TemporaryDirectory
from pdf2image import convert_from_path
import easyocr
from PIL import Image
def OCRFINAL(pdf_name, output_file, out_directory=Path("~").expanduser(), dpi=200):
    # Initialize EasyOCR reader (English language assumed)
    reader = easyocr.Reader(['en'])
    
    PDF_file = Path(pdf_name)
    image_file_list = []
    text_file = out_directory / Path(output_file)

    with TemporaryDirectory() as tempdir:
        pdf_pages = convert_from_path(PDF_file, dpi=dpi, poppler_path="C:\\Users\\Prakhar Agrawal\\Downloads\\Release-24.08.0-0\\poppler-24.08.0\\Library\\bin")
        print("pdf_pages", pdf_pages)
        for page_enumeration, page in enumerate(pdf_pages, start=1):
            filename = f"{tempdir}/page_{page_enumeration:03}.jpg"  # Corrected path separator for cross-platform compatibility
            page.save(filename, "JPEG")
            image_file_list.append(filename)

        # Extract text using EasyOCR
        with open(text_file, "a") as output_file:
            for image_file in image_file_list:
                # Read text from the image using EasyOCR
                text = " ".join(reader.readtext(image_file, detail=0))
                text = text.replace("-\n", "")
                output_file.write(text)

        # Read the whole extracted text
        with open(text_file, "r") as f:
            textFinal = f.read()

        # Split text into paragraphs of 150 words each
        paragraphs = []
        words = textFinal.split()
        for i in range(0, len(words), 150):
            paragraphs.append(' '.join(words[i:i + 150]))

        # Delete the text file after processing
        if os.path.exists(text_file):
            os.remove(text_file)

    return paragraphs

ans = OCRFINAL("VivekKumarMishraResume_13.pdf", "output.txt" , out_directory=Path("assets/"))
print(ans)