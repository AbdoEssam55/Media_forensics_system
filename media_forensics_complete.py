import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torchaudio
import torchaudio.transforms as T

import librosa
import numpy as np
import cv2
import soundfile as sf

from pathlib import Path
import os
import subprocess
import tempfile
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")


# ============================================================================
# CONFIGURATION
# ============================================================================

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
    
    # Loss and optimization
    WEIGHT_DECAY = 1e-5
    DROPOUT_RATE = 0.3
    CONFIDENCE_THRESHOLD = 0.5


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def load_image(image_path: str, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    image = Image.open(image_path).convert('RGB')
    image = image.resize(size, Image.Resampling.LANCZOS)
    return np.array(image)

def load_audio(audio_path: str, sr: int = 16000, duration: float = None) -> Tuple[np.ndarray, int]:
    try:
        y, sr_loaded = librosa.load(audio_path, sr=sr, duration=duration)
        return y, sr
    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        return None, sr

def extract_video_frames(video_path: str, sample_rate: int = 10, max_frames: int = 30) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_count += 1
        if len(frames) >= max_frames:
            break
    
    cap.release()
    return frames

def extract_audio_from_video(video_path: str) -> Optional[str]:
    try:
        output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        cmd = f'ffmpeg -i {video_path} -q:a 9 -n {output_path}'
        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=30)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        return None
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class ImageFeatureExtractor:
    
    @staticmethod
    def extract_frequency_features(image: np.ndarray) -> Dict[str, float]:
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # FFT analysis
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # DCT analysis
        dct = cv2.dct(gray.astype(np.float32))
        dct_magnitude = np.abs(dct)
        
        features = {
            'fft_mean': float(np.mean(magnitude)),
            'fft_std': float(np.std(magnitude)),
            'fft_entropy': float(-np.sum(magnitude * np.log(magnitude + 1e-10))),
            'dct_mean': float(np.mean(dct_magnitude)),
            'dct_std': float(np.std(dct_magnitude)),
            'dct_entropy': float(-np.sum(dct_magnitude * np.log(dct_magnitude + 1e-10)))
        }
        
        return features


class AudioFeatureExtractor:
    
    @staticmethod
    def extract_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfcc
    
    @staticmethod
    def extract_mel_spectrogram(y: np.ndarray, sr: int, n_mels: int = 64) -> np.ndarray:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db
    
    @staticmethod
    def extract_spectral_features(y: np.ndarray, sr: int) -> Dict[str, float]:
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),
            'spectral_centroid_std': float(np.std(spectral_centroid)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_rolloff_std': float(np.std(spectral_rolloff)),
            'zcr_mean': float(np.mean(zero_crossing_rate)),
            'zcr_std': float(np.std(zero_crossing_rate))
        }


# ============================================================================
# PYTORCH MODEL ARCHITECTURES
# ============================================================================

class ImageDetectorModel(nn.Module):
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3):
        super(ImageDetectorModel, self).__init__()
        
        self.backbone = models.resnet50(pretrained=True)
        
        # Freeze early layers (transfer learning)
        for param in self.backbone.layer1.parameters():
            param.requires_grad = False
        
        # Modify final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class AudioDetectorModel(nn.Module):
    
    def __init__(self, num_classes: int = 2, num_mfcc: int = 13, dropout_rate: float = 0.3):
        super(AudioDetectorModel, self).__init__()
        
        self.conv1 = nn.Conv1d(num_mfcc, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================================
# DATASET CLASSES
# ============================================================================

class ImageForensicsDataset(Dataset):
    
    def __init__(self, image_dir: str, label: int, transform=None):
        self.image_paths = list(Path(image_dir).glob('*.jpg')) + \
                          list(Path(image_dir).glob('*.png')) + \
                          list(Path(image_dir).glob('*.jpeg'))
        self.label = label
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.label


class AudioForensicsDataset(Dataset):
    
    def __init__(self, audio_dir: str, label: int, sr: int = 16000, n_mfcc: int = 13, max_duration: float = 30):
        self.audio_paths = list(Path(audio_dir).glob('*.wav')) + \
                          list(Path(audio_dir).glob('*.mp3'))
        self.label = label
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_duration = max_duration
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        y, sr = librosa.load(self.audio_paths[idx], sr=self.sr, duration=self.max_duration)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        
        # Pad or trim to fixed length (100 time steps)
        if mfcc.shape[1] < 100:
            mfcc = np.pad(mfcc, ((0, 0), (0, 100 - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :100]
        
        return torch.FloatTensor(mfcc), self.label


# ============================================================================
# IMAGE DETECTOR CLASS
# ============================================================================

class ImageDetector:
    
    def __init__(self, model_path: Optional[str] = None, device: torch.device = Config.DEVICE):
        self.device = device
        self.model = ImageDetectorModel().to(device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.model.eval()
        
        self.transform = Compose([
            Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
        ])
    
    def detect(self, image_path: str) -> Dict:
        try:
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            # Extract frequency features
            freq_features = ImageFeatureExtractor.extract_frequency_features(image_array)
            
            # Model inference
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = F.softmax(outputs, dim=1)
                fake_confidence = probabilities[0, 1].item() * 100
            
            classification = 'Fake' if fake_confidence > 50 else 'Real'
            
            return {
                'classification': classification,
                'confidence': fake_confidence,
                'frequency_features': freq_features,
                'media_type': 'image',
                'status': 'success'
            }
        
        except Exception as e:
            return {'error': str(e), 'status': 'failed'}


# ============================================================================
# AUDIO DETECTOR CLASS
# ============================================================================

class AudioDetector:
    
    def __init__(self, model_path: Optional[str] = None, device: torch.device = Config.DEVICE):
        self.device = device
        self.sr = Config.AUDIO_SAMPLE_RATE
        self.n_mfcc = Config.AUDIO_MFCC_COEFFS
        
        self.model = AudioDetectorModel(num_mfcc=self.n_mfcc).to(device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.model.eval()
    
    def detect(self, audio_path: str) -> Dict:
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, duration=Config.AUDIO_MAX_DURATION)
            
            if len(y) == 0:
                return {'error': 'Audio file is empty', 'status': 'failed'}
            
            # Extract features
            mfcc = AudioFeatureExtractor.extract_mfcc(y, sr, self.n_mfcc)
            spectral_features = AudioFeatureExtractor.extract_spectral_features(y, sr)
            
            # Prepare tensor
            if mfcc.shape[1] < 100:
                mfcc = np.pad(mfcc, ((0, 0), (0, 100 - mfcc.shape[1])), mode='constant')
            else:
                mfcc = mfcc[:, :100]
            
            tensor = torch.FloatTensor(mfcc).unsqueeze(0).to(self.device)
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = F.softmax(outputs, dim=1)
                fake_confidence = probabilities[0, 1].item() * 100
            
            classification = 'Fake' if fake_confidence > 50 else 'Real'
            
            return {
                'classification': classification,
                'confidence': fake_confidence,
                'spectral_features': spectral_features,
                'media_type': 'audio',
                'status': 'success'
            }
        
        except Exception as e:
            return {'error': str(e), 'status': 'failed'}


# ============================================================================
# VIDEO DETECTOR CLASS
# ============================================================================

class VideoDetector:
    
    def __init__(self, image_model_path: Optional[str] = None,
                 audio_model_path: Optional[str] = None,
                 device: torch.device = Config.DEVICE):
        self.device = device
        self.image_detector = ImageDetector(image_model_path, device)
        self.audio_detector = AudioDetector(audio_model_path, device)
    
    def analyze_temporal_consistency(self, frames: List[np.ndarray]) -> float:
        if len(frames) < 2:
            return 0.5
        
        differences = []
        
        for i in range(1, min(len(frames), 15)):
            frame1_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            frame2_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            diff = cv2.absdiff(frame1_gray, frame2_gray)
            differences.append(np.mean(diff))
        
        if not differences:
            return 0.5
        
        motion_variance = np.std(differences)
        temporal_score = 1.0 - min(motion_variance / 100.0, 1.0)
        
        return float(temporal_score)
    
    def detect(self, video_path: str, sample_rate: int = 10, 
               fusion_weight_visual: float = 0.6) -> Dict:
        try:
            # Extract frames
            frames = extract_video_frames(video_path, sample_rate, Config.VIDEO_MAX_FRAMES)
            
            if not frames:
                return {'error': 'Could not extract frames', 'status': 'failed'}
            
            # Visual analysis
            visual_confidences = []
            
            for frame in frames[:Config.VIDEO_MAX_FRAMES]:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else frame
                
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    cv2.imwrite(tmp.name, frame)
                    result = self.image_detector.detect(tmp.name)
                    
                    if result.get('status') == 'success':
                        visual_confidences.append(result['confidence'])
                    
                    os.unlink(tmp.name)
            
            if not visual_confidences:
                return {'error': 'No frames could be analyzed', 'status': 'failed'}
            
            avg_visual_confidence = np.mean(visual_confidences)
            temporal_score = self.analyze_temporal_consistency(frames)
            
            # Combine visual and temporal
            visual_score = avg_visual_confidence * 0.7 + temporal_score * 30
            
            # Audio analysis
            audio_path = extract_audio_from_video(video_path)
            has_audio = False
            audio_score = 50.0
            
            if audio_path:
                try:
                    audio_result = self.audio_detector.detect(audio_path)
                    if audio_result.get('status') == 'success':
                        audio_score = audio_result['confidence']
                        has_audio = True
                finally:
                    if os.path.exists(audio_path):
                        os.unlink(audio_path)
            
            # Fusion logic
            if has_audio:
                final_confidence = (visual_score * fusion_weight_visual + 
                                  audio_score * (1 - fusion_weight_visual))
            else:
                final_confidence = visual_score
            
            classification = 'Fake' if final_confidence > 50 else 'Real'
            
            return {
                'classification': classification,
                'final_confidence': min(final_confidence, 100.0),
                'visual_score': min(visual_score, 100.0),
                'audio_score': audio_score if has_audio else None,
                'has_audio': has_audio,
                'temporal_consistency': temporal_score,
                'frames_analyzed': len(visual_confidences),
                'media_type': 'video',
                'status': 'success'
            }
        
        except Exception as e:
            return {'error': str(e), 'status': 'failed'}


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_image_detector(train_loader: DataLoader, val_loader: DataLoader,
                        epochs: int = 25, lr: float = 1e-4,
                        device: torch.device = Config.DEVICE) -> Dict:
    
    model = ImageDetectorModel().to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        train_accuracy = 100 * train_correct / train_total
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_accuracy)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_accuracy = 100 * val_correct / val_total
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_accuracy)
        
        scheduler.step(val_accuracy)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {history['train_loss'][-1]:.4f} | "
                  f"Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")
    
    return model, history


def train_audio_detector(train_loader: DataLoader, val_loader: DataLoader,
                        epochs: int = 25, lr: float = 1e-3,
                        device: torch.device = Config.DEVICE) -> Dict:
    
    model = AudioDetectorModel().to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for mfcc, labels in train_loader:
            mfcc, labels = mfcc.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(mfcc)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_f1 = f1_score(train_labels, train_preds)
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_f1'].append(train_f1)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for mfcc, labels in val_loader:
                mfcc, labels = mfcc.to(device), labels.to(device)
                outputs = model(mfcc)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(val_labels, val_preds)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_f1'].append(val_f1)
        
        scheduler.step(val_f1)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {history['train_loss'][-1]:.4f} | "
                  f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
    
    return model, history


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_detector(detector, test_paths: List[str], test_labels: List[int],
                     detector_type: str = 'image') -> Dict:
    
    predictions = []
    confidences = []
    correct = 0
    
    for path, true_label in zip(test_paths, test_labels):
        result = detector.detect(path)
        
        if result.get('status') == 'success':
            predicted_label = 0 if result['classification'] == 'Real' else 1
            predictions.append(predicted_label)
            confidences.append(result['confidence'])
            
            if predicted_label == true_label:
                correct += 1
    
    accuracy = correct / len(predictions) if predictions else 0
    precision = precision_score(test_labels[:len(predictions)], predictions, zero_division=0)
    recall = recall_score(test_labels[:len(predictions)], predictions, zero_division=0)
    f1 = f1_score(test_labels[:len(predictions)], predictions, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': predictions,
        'confidences': confidences
    }


# ============================================================================
# DEMO: Complete Inference Example
# ============================================================================

def demo_inference():
    print("Initializing detectors...")
    
    image_detector = ImageDetector()
    audio_detector = AudioDetector()
    video_detector = VideoDetector()
    
    print("Detectors initialized successfully")
    print(f"Device: {Config.DEVICE}")
    print(f"Image Size: {Config.IMAGE_SIZE}")
    print(f"Audio Sample Rate: {Config.AUDIO_SAMPLE_RATE}")
    
    # Example: Process image
    print("\n--- Image Detection Example ---")
    print("To analyze an image: detector.detect('path/to/image.jpg')")
    print("Output: {'classification': 'Real'/'Fake', 'confidence': 0-100, ...}")
    
    # Example: Process audio
    print("\n--- Audio Detection Example ---")
    print("To analyze audio: detector.detect('path/to/audio.wav')")
    print("Output: {'classification': 'Real'/'Fake', 'confidence': 0-100, ...}")
    
    # Example: Process video
    print("\n--- Video Detection Example ---")
    print("To analyze video: detector.detect('path/to/video.mp4')")
    print("Output: {'classification': 'Real'/'Fake', 'final_confidence': 0-100,")
    print("         'visual_score': 0-100, 'audio_score': 0-100, ...}")
    
    print("\n--- Scene-Agnostic Analysis ---")
    print("Video detector works on:")
    print("  - Landscapes and natural scenes")
    print("  - Urban and architectural footage")
    print("  - Cinematic environments")
    print("  - Time-lapse and abstract videos")
    print("  - Audio-less videos (visual analysis only)")
    
    return image_detector, audio_detector, video_detector


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    set_seed(Config.SEED)
    
    # Initialize all detectors
    image_det, audio_det, video_det = demo_inference()
    
    print("\n" + "="*70)
    print("MEDIA FORENSICS SYSTEM - READY FOR USE")
    print("="*70)
    print("\nAll detector classes are initialized and ready for inference.")
    print("\nSupported media types:")
    print("  - Images: PNG, JPG, JPEG, BMP, TIFF")
    print("  - Audio: WAV, MP3, OGG, FLAC")
    print("  - Video: MP4, MKV, AVI, MOV")
    print("\nTo use in your own code:")
    print("  from media_forensics import ImageDetector, AudioDetector, VideoDetector")
    print("  detector = ImageDetector('models/image_detector.pth')")
    print("  result = detector.detect('image.jpg')")
