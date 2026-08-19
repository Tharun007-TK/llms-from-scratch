from huggingface_hub import HfApi

api = HfApi()

print("Uploading to Hugging Face Hub (Tharun007/FinGPT-131M)...")
try:
    api.upload_folder(
        folder_path="./fingpt-131m",
        repo_id="Tharun007/FinGPT-131M",
        repo_type="model",
    )
    print("Upload complete!")
except Exception as e:
    print(f"Failed to upload: {e}")
    print("\nYou might need to authenticate. Run `huggingface-cli login` in your terminal and paste your Hugging Face write token.")
