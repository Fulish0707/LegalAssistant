import pytesseract
from PIL import Image
import os

# Optional: If Tesseract is not in your PATH, you might need to set the command path
# For macOS, if installed via Homebrew, it's usually in /usr/local/bin/tesseract or /opt/homebrew/bin/tesseract
# For Windows, it's typically C:\Program Files\Tesseract-OCR\tesseract.exe
# Example:
# if os.name == 'posix': # For macOS/Linux
#     # Check common paths for Tesseract
#     if os.path.exists('/opt/homebrew/bin/tesseract'):
#         pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
#     elif os.path.exists('/usr/local/bin/tesseract'):
#         pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
#     # Add other paths if necessary
# elif os.name == 'nt': # For Windows
#     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_text(image_path):
    """
    Extracts text from an image file using Tesseract OCR.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The extracted text, or an error message string if extraction fails.
    """
    if not os.path.exists(image_path):
        return "Error: Image file not found."

    try:
        # Open the image file
        img = Image.open(image_path)
        
        # Perform OCR using pytesseract
        # You can specify language if needed, e.g., lang='eng+kor' for English and Korean
        text = pytesseract.image_to_string(img)
        return text
    except pytesseract.TesseractNotFoundError:
        return ("Error: Tesseract is not installed or not found in your PATH. "
                "Please install Tesseract OCR and ensure it's accessible. "
                "You might also need to set pytesseract.pytesseract.tesseract_cmd "
                "at the beginning of this script if Tesseract is installed in a non-standard location.")
    except Exception as e:
        return f"Error during OCR processing: {str(e)}"

if __name__ == '__main__':
    # This block is for testing the script directly.
    # To use it:
    # 1. Make sure you have an image file (e.g., 'test_image.png') in the same directory
    #    or provide the full path to an image.
    # 2. Uncomment the lines below and run: python llm_ocr.py
    
    # print("llm_ocr.py script is being run directly for testing.")
    # test_image_filename = "test_image.png" # Replace with your test image file
    
    # # Create a dummy image for testing if you don't have one readily available
    # try:
    #     from PIL import Image, ImageDraw, ImageFont
    #     if not os.path.exists(test_image_filename):
    #         print(f"Creating a dummy test image: {test_image_filename}")
    #         try:
    #             img_test = Image.new('RGB', (400, 100), color = (255, 255, 255))
    #             d = ImageDraw.Draw(img_test)
    #             # Attempt to load a font; falls back to a default if specific font not found
    #             try:
    #                 font = ImageFont.truetype("arial.ttf", 20)
    #             except IOError:
    #                 font = ImageFont.load_default()
    #             d.text((10,10), "Hello Tesseract OCR", fill=(0,0,0), font=font)
    #             img_test.save(test_image_filename)
    #             print(f"Dummy image '{test_image_filename}' created successfully.")
    #         except Exception as e_img:
    #             print(f"Could not create dummy image: {e_img}. Please provide your own test image.")
    #             test_image_filename = None # Ensure test doesn't run if image creation failed
    # except ImportError:
    #     print("Pillow (PIL) is not installed. Cannot create a dummy image. Please provide your own test image.")
    #     test_image_filename = None

    # if test_image_filename and os.path.exists(test_image_filename):
    #     print(f"Attempting to OCR '{test_image_filename}'...")
    #     extracted_text = detect_text(test_image_filename)
    #     print(f"\n--- Extracted Text from '{test_image_filename}' ---")
    #     print(extracted_text)
    #     print("--- End of Extracted Text ---")
    # elif test_image_filename:
    #     print(f"Test image '{test_image_filename}' not found or could not be created. Skipping example run.")
    # else:
    #     print("No test image available. Skipping example run.")
        
    print("\nllm_ocr.py loaded. The detect_text function is ready to be used by app.py.")
    print("Ensure Tesseract OCR is installed and 'pytesseract' and 'Pillow' Python packages are installed.")
