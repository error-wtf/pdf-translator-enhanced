"""
Download Marker models for PDF extraction.
This script downloads ~2GB of AI models from HuggingFace.

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
import sys
import os

# Set HuggingFace cache to a predictable location
HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

def check_models_exist():
    """Check if Marker models are already downloaded."""
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        cached_repos = [r.repo_id for r in cache.repos]
        required = ["vikp/surya_det3", "vikp/surya_rec2", "vikp/texify"]
        found = sum(1 for r in required if r in cached_repos)
        return found >= 2
    except Exception:
        return False

def main():
    print("=" * 60)
    print("  Marker Models for Scientific PDF Extraction")
    print("=" * 60)
    print()
    
    # Check if already downloaded
    if check_models_exist():
        print("Marker models already downloaded!")
        print(f"Cache location: {HF_CACHE}")
        print()
        print("=" * 60)
        print("  SKIPPED: Models already available")
        print("=" * 60)
        return 0
    
    print("Downloading ~2GB of AI models from HuggingFace...")
    print("This may take 5-10 minutes on first run.")
    print()
    print(f"Cache location: {HF_CACHE}")
    print()
    
    try:
        print("[1/3] Importing Marker...")
        from marker.models import create_model_dict
        
        print("[2/3] Downloading models...")
        print("      - surya_det3 (detection)")
        print("      - surya_rec2 (recognition)")
        print("      - texify (LaTeX)")
        print("      - surya_layout (layout)")
        print("      - surya_tablerec (tables)")
        print()
        
        models = create_model_dict()
        
        print()
        print("[3/3] Verifying models...")
        print(f"      Loaded {len(models)} model components")
        
        print()
        print("=" * 60)
        print("  SUCCESS: Marker models downloaded!")
        print(f"  Location: {HF_CACHE}")
        print("=" * 60)
        return 0
        
    except ImportError as e:
        print(f"ERROR: Marker not installed: {e}")
        print("Run: pip install marker-pdf huggingface_hub")
        return 1
    except Exception as e:
        print(f"ERROR: Download failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
