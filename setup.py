#!/usr/bin/env python3
"""
Setup script for Media Forensics Detection System
Handles environment initialization, model downloads, and verification
"""

import os
import sys
import subprocess
import torch
import argparse
from pathlib import Path


def check_python_version():
    """Verify Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"ERROR: Python 3.8+ required (found {version.major}.{version.minor})")
        sys.exit(1)
    print(f"OK: Python {version.major}.{version.minor}.{version.micro}")


def check_pytorch():
    """Verify PyTorch installation"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device = "GPU (CUDA)" if cuda_available else "CPU"
        print(f"OK: PyTorch {torch.__version__} ({device})")
        if cuda_available:
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
        return cuda_available
    except ImportError:
        print("ERROR: PyTorch not installed")
        return False


def check_dependencies():
    """Verify all required packages are installed"""
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'torchaudio': 'TorchAudio',
        'librosa': 'librosa',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'sklearn': 'scikit-learn',
        'soundfile': 'soundfile',
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"OK: {name}")
        except ImportError:
            print(f"MISSING: {name}")
            missing.append(module)
    
    return len(missing) == 0, missing


def check_ffmpeg():
    """Verify ffmpeg is installed"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.decode().split('\n')[0]
            print(f"OK: ffmpeg ({version})")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    print("WARNING: ffmpeg not found (required for video audio extraction)")
    print("  Install: sudo apt-get install ffmpeg  (Linux)")
    print("           brew install ffmpeg          (macOS)")
    print("           https://ffmpeg.org/download.html (Windows)")
    return False


def create_directories():
    """Create necessary project directories"""
    dirs = [
        'models',
        'data/images/real',
        'data/images/fake',
        'data/audio/real',
        'data/audio/fake',
        'data/videos/real',
        'data/videos/fake',
        'notebooks',
        'logs',
        'results'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"OK: Created {dir_path}")


def download_pretrained_models():
    """Download pretrained models from HuggingFace or GitHub"""
    model_urls = {
        'image_detector.pth': 'https://example.com/models/image_detector.pth',
        'audio_detector.pth': 'https://example.com/models/audio_detector.pth',
    }
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    for model_name, url in model_urls.items():
        model_path = models_dir / model_name
        
        if model_path.exists():
            print(f"OK: {model_name} already exists")
            continue
        
        print(f"Downloading {model_name}...")
        print(f"  Note: Update the URL in setup.py to actual model location")
        print(f"  Expected location: {model_path}")


def generate_config():
    """Generate config.py file if it doesn't exist"""
    config_content = '''"""
Configuration for Media Forensics Detection System
"""

import torch
from pathlib import Path

class Config:
    # General
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    
    # Image detector
    IMAGE_SIZE = 224
    IMAGE_BATCH_SIZE = 32
    IMAGE_EPOCHS = 25
    IMAGE_LR = 1e-4
    
    # Audio detector
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_MFCC_COEFFS = 13
    AUDIO_MEL_BINS = 64
    AUDIO_FRAME_LENGTH = 2048
    AUDIO_HOP_LENGTH = 512
    AUDIO_MAX_DURATION = 30  # seconds
    AUDIO_BATCH_SIZE = 16
    AUDIO_EPOCHS = 25
    AUDIO_LR = 1e-3
    
    # Video detector
    VIDEO_SAMPLE_FRAME_RATE = 10  # sample every Nth frame
    VIDEO_MAX_FRAMES = 30
    VIDEO_VISUAL_WEIGHT = 0.6
    VIDEO_AUDIO_WEIGHT = 0.4
    
    # Paths
    MODEL_DIR = Path('models')
    DATA_DIR = Path('data')
    RESULTS_DIR = Path('results')
    
    # Loss and optimization
    WEIGHT_DECAY = 1e-5
    DROPOUT_RATE = 0.3
    CONFIDENCE_THRESHOLD = 0.5
    
    # Logging
    LOG_DIR = Path('logs')
    VERBOSE = True
'''
    
    config_path = Path('config.py')
    if not config_path.exists():
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"OK: Generated {config_path}")
    else:
        print(f"OK: {config_path} already exists")


def test_imports():
    """Test that main modules can be imported"""
    try:
        print("\nTesting imports...")
        from media_forensics_complete import (
            ImageDetector, AudioDetector, VideoDetector,
            ImageDetectorModel, AudioDetectorModel,
            ImageFeatureExtractor, AudioFeatureExtractor,
            Config
        )
        print("OK: All modules imported successfully")
        return True
    except Exception as e:
        print(f"ERROR: Failed to import modules: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Setup Media Forensics Detection System'
    )
    parser.add_argument('--skip-models', action='store_true',
                       help='Skip model download')
    parser.add_argument('--test', action='store_true',
                       help='Run tests only')
    args = parser.parse_args()
    
    print("="*70)
    print("MEDIA FORENSICS DETECTION SYSTEM - SETUP")
    print("="*70)
    
    print("\n1. Checking Python version...")
    check_python_version()
    
    print("\n2. Checking PyTorch installation...")
    cuda_available = check_pytorch()
    
    print("\n3. Checking dependencies...")
    all_ok, missing = check_dependencies()
    
    if not all_ok:
        print("\nERROR: Missing dependencies!")
        print("Install with: pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n4. Checking ffmpeg...")
    check_ffmpeg()
    
    if not args.test:
        print("\n5. Creating directories...")
        create_directories()
        
        if not args.skip_models:
            print("\n6. Downloading pretrained models...")
            download_pretrained_models()
        
        print("\n7. Generating configuration...")
        generate_config()
    
    print("\n8. Testing imports...")
    if test_imports():
        print("\n" + "="*70)
        print("SETUP COMPLETE - System ready for use")
        print("="*70)
        print("\nNext steps:")
        print("  1. Launch web UI: streamlit run app_streamlit.py")
        print("  2. Start API: python -m uvicorn app_fastapi:app --reload")
        print("  3. Run notebooks: jupyter notebook")
        print("\nDocumentation: See README.md")
    else:
        print("\nSetup completed with warnings. Check import errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
