"""
Marker Worker - Runs Marker in a separate process to avoid Gradio blocking.

Usage: python marker_worker.py input.pdf output_dir
"""
import sys
import json
from pathlib import Path

def run_marker(pdf_path: str, output_dir: str) -> dict:
    """Run Marker conversion and return results as JSON."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
        print(f"MARKER_STATUS: PyTorch {torch.__version__}, CUDA: {cuda_available}, Device: {device}", flush=True)
        
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
        
        print("MARKER_STATUS: Loading models...", flush=True)
        converter = PdfConverter(artifact_dict=create_model_dict())
        
        print("MARKER_STATUS: Converting PDF...", flush=True)
        rendered = converter(pdf_path)
        text, _, images = text_from_rendered(rendered)
        
        # Save markdown
        output_path = Path(output_dir) / "extracted.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Save images
        for img_name, img_data in images.items():
            img_path = Path(output_dir) / img_name
            with open(img_path, 'wb') as f:
                f.write(img_data)
        
        print("MARKER_STATUS: Complete!", flush=True)
        return {
            "success": True,
            "output_path": str(output_path),
            "text_length": len(text),
            "image_count": len(images)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"success": False, "error": "Usage: marker_worker.py input.pdf output_dir"}))
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    result = run_marker(pdf_path, output_dir)
    print(f"MARKER_RESULT: {json.dumps(result)}", flush=True)
    sys.exit(0 if result["success"] else 1)
