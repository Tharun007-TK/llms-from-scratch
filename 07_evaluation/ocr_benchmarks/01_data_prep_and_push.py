import os
import random
import fitz  # PyMuPDF
from datasets import Dataset
from pathlib import Path
from tqdm import tqdm
import io
from PIL import Image as PILImage

def prep_and_push_dataset():
    """
    Samples 1000 PDFs, extracts the first up to 3 pages as images,
    compiles them into a Hugging Face Dataset, and pushes to the hub.
    """
    # Setup paths
    raw_dir = Path("c:/Projectx/LLMs-from-scratch/raw/aar_rulings")
    if not raw_dir.exists():
        print(f"Error: {raw_dir} does not exist.")
        return
        
    pdf_files = list(raw_dir.glob("*.pdf"))
    if len(pdf_files) == 0:
        print("No PDF files found.")
        return

    # Sample 1000 PDFs (or all if less than 1000)
    sample_size = min(1000, len(pdf_files))
    sampled_pdfs = random.sample(pdf_files, sample_size)
    print(f"Sampled {sample_size} documents.")

    def image_generator():
        for pdf_path in tqdm(sampled_pdfs, desc="Extracting images"):
            try:
                doc = fitz.open(pdf_path)
                for page_num in range(min(3, len(doc))):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=150) # Moderate DPI for OCR
                    
                    # Convert directly to PIL Image
                    img_bytes = pix.tobytes("png")
                    pil_img = PILImage.open(io.BytesIO(img_bytes))
                    
                    yield {
                        "image": pil_img, 
                        "pdf_name": pdf_path.name,
                        "page_num": page_num
                    }
                doc.close()
            except Exception as e:
                print(f"Failed to process {pdf_path.name}: {e}")
            
    # Create Hugging Face Dataset using a generator to avoid memory/pyarrow offset limits
    print("Creating Hugging Face Dataset object...")
    dataset = Dataset.from_generator(image_generator)
    
    # Push to hub
    # Note: Requires huggingface-cli login or HF_TOKEN env var to be set
    dataset_name = "Tharun007/aar_rulings_ocr_sample"
    print(f"\nPushing dataset to {dataset_name}...")
    try:
        dataset.push_to_hub(dataset_name, private=True)
        print("Push successful!")
    except Exception as e:
        print(f"\nFailed to push to hub. Did you set your HF_TOKEN? Error: {e}")
        print(f"To run this locally, use: huggingface-cli login")

if __name__ == "__main__":
    prep_and_push_dataset()
