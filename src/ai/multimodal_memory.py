"""Multimodal Memory Module for StudSar V4

Supports images, audio, and video alongside text with cross-modal associations
and semantic understanding across different data types.
"""

import asyncio
import hashlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import base64
from datetime import datetime

# Image processing
try:
    from PIL import Image, ImageOps
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Audio processing
try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Video processing
try:
    import cv2
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False

# Computer vision
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import CLIPProcessor, CLIPModel
    import torch
    VISION_TRANSFORMERS_AVAILABLE = True
except ImportError:
    VISION_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """Supported modality types"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"

@dataclass
class MediaMetadata:
    """Metadata for media files"""
    file_path: str
    file_size: int
    mime_type: str
    duration: Optional[float] = None  # For audio/video
    dimensions: Optional[Tuple[int, int]] = None  # For images/video
    sample_rate: Optional[int] = None  # For audio
    fps: Optional[float] = None  # For video
    created_at: datetime = field(default_factory=datetime.now)
    checksum: Optional[str] = None

@dataclass
class ModalityFeatures:
    """Features extracted from different modalities"""
    modality: ModalityType
    features: List[float]
    metadata: Dict[str, Any]
    confidence: float
    description: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    embeddings: Optional[List[float]] = None

@dataclass
class MultimodalMarker:
    """Enhanced marker supporting multiple modalities"""
    id: str
    primary_content: str  # Text content or description
    modalities: Dict[ModalityType, Any]  # Content for each modality
    features: Dict[ModalityType, ModalityFeatures]  # Extracted features
    cross_modal_links: List[str] = field(default_factory=list)  # Links to related markers
    metadata: MediaMetadata = None
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    reputation: float = 0.0

class BaseModalityProcessor(ABC):
    """Abstract base class for modality processors"""
    
    @abstractmethod
    async def extract_features(self, content: Any) -> ModalityFeatures:
        """Extract features from content"""
        pass
    
    @abstractmethod
    async def generate_description(self, content: Any) -> str:
        """Generate text description of content"""
        pass
    
    @abstractmethod
    def is_supported(self, file_path: str) -> bool:
        """Check if file type is supported"""
        pass

class ImageProcessor(BaseModalityProcessor):
    """Image processing and feature extraction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.clip_model = None
        self.clip_processor = None
        self.blip_model = None
        self.blip_processor = None
        
        if VISION_TRANSFORMERS_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize vision models"""
        try:
            # CLIP for image embeddings
            model_name = self.config.get('clip_model', 'openai/clip-vit-base-patch32')
            self.clip_model = CLIPModel.from_pretrained(model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            
            # BLIP for image captioning
            caption_model = self.config.get('blip_model', 'Salesforce/blip-image-captioning-base')
            self.blip_model = BlipForConditionalGeneration.from_pretrained(caption_model)
            self.blip_processor = BlipProcessor.from_pretrained(caption_model)
            
            logger.info("Vision models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vision models: {e}")
    
    async def extract_features(self, image_path: str) -> ModalityFeatures:
        """Extract features from image"""
        try:
            if not PIL_AVAILABLE:
                raise ImportError("PIL not available")
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Extract basic features
            width, height = image.size
            aspect_ratio = width / height
            
            # Generate embeddings if CLIP is available
            embeddings = []
            if self.clip_model is not None:
                inputs = self.clip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    embeddings = image_features.squeeze().tolist()
            
            # Extract color histogram
            color_features = self._extract_color_features(image)
            
            # Combine features
            features = [width, height, aspect_ratio] + color_features + embeddings
            
            # Generate description
            description = await self.generate_description(image_path)
            
            return ModalityFeatures(
                modality=ModalityType.IMAGE,
                features=features,
                metadata={
                    'width': width,
                    'height': height,
                    'aspect_ratio': aspect_ratio,
                    'color_channels': len(image.getbands())
                },
                confidence=0.9 if self.clip_model else 0.6,
                description=description,
                keywords=self._extract_visual_keywords(description),
                embeddings=embeddings
            )
            
        except Exception as e:
            logger.error(f"Image feature extraction failed: {e}")
            return self._fallback_image_features(image_path)
    
    async def generate_description(self, image_path: str) -> str:
        """Generate text description of image"""
        try:
            if self.blip_model is None:
                return f"Image file: {os.path.basename(image_path)}"
            
            image = Image.open(image_path).convert('RGB')
            inputs = self.blip_processor(image, return_tensors="pt")
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
                description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            return description
            
        except Exception as e:
            logger.error(f"Image description generation failed: {e}")
            return f"Image file: {os.path.basename(image_path)}"
    
    def is_supported(self, file_path: str) -> bool:
        """Check if image file is supported"""
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        return Path(file_path).suffix.lower() in supported_extensions
    
    def _extract_color_features(self, image: Image.Image) -> List[float]:
        """Extract color histogram features"""
        try:
            # Convert to RGB and resize for efficiency
            image = image.resize((64, 64))
            
            # Extract color histogram
            hist_r = image.histogram()[0:256]
            hist_g = image.histogram()[256:512]
            hist_b = image.histogram()[512:768]
            
            # Normalize histograms
            total_pixels = 64 * 64
            hist_r = [h / total_pixels for h in hist_r[::8]]  # Sample every 8th bin
            hist_g = [h / total_pixels for h in hist_g[::8]]
            hist_b = [h / total_pixels for h in hist_b[::8]]
            
            return hist_r + hist_g + hist_b
            
        except Exception as e:
            logger.error(f"Color feature extraction failed: {e}")
            return [0.0] * 96  # 32 bins per channel
    
    def _extract_visual_keywords(self, description: str) -> List[str]:
        """Extract keywords from image description"""
        if not description:
            return []
        
        # Simple keyword extraction
        words = description.lower().split()
        # Filter out common words
        stop_words = {'a', 'an', 'the', 'is', 'are', 'with', 'of', 'in', 'on', 'at'}
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords[:5]
    
    def _fallback_image_features(self, image_path: str) -> ModalityFeatures:
        """Fallback features when processing fails"""
        return ModalityFeatures(
            modality=ModalityType.IMAGE,
            features=[0.0] * 100,  # Dummy features
            metadata={'file_path': image_path},
            confidence=0.1,
            description=f"Image file: {os.path.basename(image_path)}",
            keywords=[]
        )

class AudioProcessor(BaseModalityProcessor):
    """Audio processing and feature extraction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_rate = config.get('sample_rate', 22050)
        self.max_duration = config.get('max_duration', 30)  # seconds
    
    async def extract_features(self, audio_path: str) -> ModalityFeatures:
        """Extract features from audio"""
        try:
            if not AUDIO_AVAILABLE:
                raise ImportError("Librosa not available")
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.max_duration)
            
            # Extract audio features
            features = []
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.extend(np.mean(mfccs, axis=1).tolist())
            features.extend(np.std(mfccs, axis=1).tolist())
            
            # Chroma features
            chroma = librosa.feature.chroma(y=y, sr=sr)
            features.extend(np.mean(chroma, axis=1).tolist())
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features.extend([np.mean(zcr), np.std(zcr)])
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.append(tempo)
            
            # Duration and energy
            duration = len(y) / sr
            energy = np.sum(y ** 2) / len(y)
            features.extend([duration, energy])
            
            description = await self.generate_description(audio_path)
            
            return ModalityFeatures(
                modality=ModalityType.AUDIO,
                features=features,
                metadata={
                    'duration': duration,
                    'sample_rate': sr,
                    'tempo': tempo,
                    'energy': energy
                },
                confidence=0.8,
                description=description,
                keywords=self._extract_audio_keywords(audio_path, features)
            )
            
        except Exception as e:
            logger.error(f"Audio feature extraction failed: {e}")
            return self._fallback_audio_features(audio_path)
    
    async def generate_description(self, audio_path: str) -> str:
        """Generate text description of audio"""
        try:
            if not AUDIO_AVAILABLE:
                return f"Audio file: {os.path.basename(audio_path)}"
            
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=5)  # Short sample
            
            # Basic audio analysis
            duration = len(y) / sr
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Classify audio type based on features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            
            if tempo > 120:
                tempo_desc = "fast-paced"
            elif tempo > 80:
                tempo_desc = "moderate-paced"
            else:
                tempo_desc = "slow-paced"
            
            if spectral_centroid > 3000:
                tone_desc = "bright"
            elif spectral_centroid > 1500:
                tone_desc = "balanced"
            else:
                tone_desc = "warm"
            
            return f"{tempo_desc} {tone_desc} audio, {duration:.1f} seconds, {tempo:.0f} BPM"
            
        except Exception as e:
            logger.error(f"Audio description generation failed: {e}")
            return f"Audio file: {os.path.basename(audio_path)}"
    
    def is_supported(self, file_path: str) -> bool:
        """Check if audio file is supported"""
        supported_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
        return Path(file_path).suffix.lower() in supported_extensions
    
    def _extract_audio_keywords(self, audio_path: str, features: List[float]) -> List[str]:
        """Extract keywords based on audio features"""
        keywords = []
        
        # Add file type
        ext = Path(audio_path).suffix.lower()
        if ext:
            keywords.append(ext[1:])  # Remove dot
        
        # Add descriptive keywords based on features
        if len(features) > 40:  # Ensure we have enough features
            tempo = features[-3] if len(features) > 2 else 0
            energy = features[-1] if len(features) > 0 else 0
            
            if tempo > 120:
                keywords.append("upbeat")
            elif tempo < 80:
                keywords.append("slow")
            
            if energy > 0.01:
                keywords.append("energetic")
            else:
                keywords.append("quiet")
        
        return keywords[:5]
    
    def _fallback_audio_features(self, audio_path: str) -> ModalityFeatures:
        """Fallback features when processing fails"""
        return ModalityFeatures(
            modality=ModalityType.AUDIO,
            features=[0.0] * 50,  # Dummy features
            metadata={'file_path': audio_path},
            confidence=0.1,
            description=f"Audio file: {os.path.basename(audio_path)}",
            keywords=[]
        )

class VideoProcessor(BaseModalityProcessor):
    """Video processing and feature extraction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_frames = config.get('max_frames', 30)
        self.image_processor = ImageProcessor(config.get('image_config', {}))
        self.audio_processor = AudioProcessor(config.get('audio_config', {}))
    
    async def extract_features(self, video_path: str) -> ModalityFeatures:
        """Extract features from video"""
        try:
            if not VIDEO_AVAILABLE:
                raise ImportError("OpenCV not available")
            
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Extract frames for analysis
            frames = []
            frame_interval = max(1, frame_count // self.max_frames)
            
            for i in range(0, frame_count, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                if len(frames) >= self.max_frames:
                    break
            
            cap.release()
            
            # Analyze frames
            visual_features = await self._analyze_frames(frames)
            
            # Basic video features
            features = [
                duration, fps, width, height, len(frames),
                width / height,  # aspect ratio
            ]
            features.extend(visual_features)
            
            description = await self.generate_description(video_path)
            
            return ModalityFeatures(
                modality=ModalityType.VIDEO,
                features=features,
                metadata={
                    'duration': duration,
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'frame_count': frame_count
                },
                confidence=0.7,
                description=description,
                keywords=self._extract_video_keywords(video_path, features)
            )
            
        except Exception as e:
            logger.error(f"Video feature extraction failed: {e}")
            return self._fallback_video_features(video_path)
    
    async def _analyze_frames(self, frames: List[np.ndarray]) -> List[float]:
        """Analyze video frames for visual features"""
        if not frames:
            return [0.0] * 20
        
        features = []
        
        # Average brightness and contrast
        brightness_values = []
        contrast_values = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            brightness_values.append(brightness)
            contrast_values.append(contrast)
        
        features.extend([
            np.mean(brightness_values),
            np.std(brightness_values),
            np.mean(contrast_values),
            np.std(contrast_values)
        ])
        
        # Motion analysis (simple frame difference)
        if len(frames) > 1:
            motion_values = []
            for i in range(1, len(frames)):
                diff = cv2.absdiff(
                    cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                )
                motion = np.mean(diff)
                motion_values.append(motion)
            
            features.extend([
                np.mean(motion_values),
                np.std(motion_values),
                max(motion_values) if motion_values else 0
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Color analysis
        color_features = []
        for frame in frames[:5]:  # Analyze first 5 frames
            for channel in range(3):  # BGR channels
                channel_mean = np.mean(frame[:, :, channel])
                color_features.append(channel_mean)
        
        # Pad or truncate to fixed size
        while len(color_features) < 15:
            color_features.append(0.0)
        features.extend(color_features[:15])
        
        return features
    
    async def generate_description(self, video_path: str) -> str:
        """Generate text description of video"""
        try:
            if not VIDEO_AVAILABLE:
                return f"Video file: {os.path.basename(video_path)}"
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            resolution = "HD" if width >= 1280 else "SD"
            duration_desc = f"{duration:.1f} seconds"
            
            return f"{resolution} video, {width}x{height}, {duration_desc}, {fps:.1f} FPS"
            
        except Exception as e:
            logger.error(f"Video description generation failed: {e}")
            return f"Video file: {os.path.basename(video_path)}"
    
    def is_supported(self, file_path: str) -> bool:
        """Check if video file is supported"""
        supported_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        return Path(file_path).suffix.lower() in supported_extensions
    
    def _extract_video_keywords(self, video_path: str, features: List[float]) -> List[str]:
        """Extract keywords based on video features"""
        keywords = []
        
        # Add file type
        ext = Path(video_path).suffix.lower()
        if ext:
            keywords.append(ext[1:])  # Remove dot
        
        # Add descriptive keywords based on features
        if len(features) > 10:
            duration = features[0]
            fps = features[1]
            width = features[2]
            height = features[3]
            
            if duration > 300:  # 5 minutes
                keywords.append("long")
            elif duration < 30:
                keywords.append("short")
            
            if width >= 1920:
                keywords.append("hd")
            elif width >= 1280:
                keywords.append("720p")
            
            if fps > 50:
                keywords.append("high-fps")
        
        return keywords[:5]
    
    def _fallback_video_features(self, video_path: str) -> ModalityFeatures:
        """Fallback features when processing fails"""
        return ModalityFeatures(
            modality=ModalityType.VIDEO,
            features=[0.0] * 30,  # Dummy features
            metadata={'file_path': video_path},
            confidence=0.1,
            description=f"Video file: {os.path.basename(video_path)}",
            keywords=[]
        )

class MultimodalMemoryManager:
    """Main manager for multimodal memory"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_path = Path(config.get('storage_path', 'multimodal_storage'))
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize processors
        self.processors = {
            ModalityType.IMAGE: ImageProcessor(config.get('image_config', {})),
            ModalityType.AUDIO: AudioProcessor(config.get('audio_config', {})),
            ModalityType.VIDEO: VideoProcessor(config.get('video_config', {}))
        }
        
        # Memory storage
        self.markers: Dict[str, MultimodalMarker] = {}
        self.cross_modal_index: Dict[str, List[str]] = {}  # keyword -> marker_ids
        
        # Load existing markers
        self._load_markers()
    
    async def add_media(self, file_path: str, description: Optional[str] = None) -> Optional[str]:
        """Add media file to multimodal memory"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            # Determine modality
            modality = self._detect_modality(file_path)
            if modality == ModalityType.TEXT:
                logger.warning(f"Text files not supported in multimodal memory: {file_path}")
                return None
            
            # Generate unique ID
            marker_id = self._generate_marker_id(file_path)
            
            # Extract features
            processor = self.processors.get(modality)
            if processor is None:
                logger.error(f"No processor available for {modality}")
                return None
            
            features = await processor.extract_features(file_path)
            
            # Create media metadata
            metadata = self._create_media_metadata(file_path)
            
            # Create multimodal marker
            marker = MultimodalMarker(
                id=marker_id,
                primary_content=description or features.description,
                modalities={modality: file_path},
                features={modality: features},
                metadata=metadata
            )
            
            # Store marker
            self.markers[marker_id] = marker
            
            # Update cross-modal index
            self._update_cross_modal_index(marker)
            
            # Save to disk
            self._save_marker(marker)
            
            logger.info(f"Added {modality.value} marker: {marker_id}")
            return marker_id
            
        except Exception as e:
            logger.error(f"Failed to add media {file_path}: {e}")
            return None
    
    async def search_multimodal(self, query: str, modalities: Optional[List[ModalityType]] = None, limit: int = 10) -> List[MultimodalMarker]:
        """Search across multiple modalities"""
        if modalities is None:
            modalities = list(ModalityType)
        
        results = []
        query_words = query.lower().split()
        
        for marker in self.markers.values():
            score = 0.0
            
            # Text matching
            if any(word in marker.primary_content.lower() for word in query_words):
                score += 1.0
            
            # Keyword matching
            for modality, features in marker.features.items():
                if modality in modalities:
                    keyword_matches = sum(1 for kw in features.keywords if any(word in kw.lower() for word in query_words))
                    score += keyword_matches * 0.5
            
            # Description matching
            for modality, features in marker.features.items():
                if modality in modalities and features.description:
                    if any(word in features.description.lower() for word in query_words):
                        score += 0.3
            
            if score > 0:
                results.append((marker, score))
        
        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return [marker for marker, score in results[:limit]]
    
    def get_marker(self, marker_id: str) -> Optional[MultimodalMarker]:
        """Get marker by ID"""
        return self.markers.get(marker_id)
    
    def list_markers(self, modality: Optional[ModalityType] = None) -> List[MultimodalMarker]:
        """List all markers, optionally filtered by modality"""
        if modality is None:
            return list(self.markers.values())
        
        return [marker for marker in self.markers.values() if modality in marker.modalities]
    
    def get_cross_modal_links(self, marker_id: str) -> List[MultimodalMarker]:
        """Get markers linked to the given marker"""
        marker = self.markers.get(marker_id)
        if not marker:
            return []
        
        linked_markers = []
        for link_id in marker.cross_modal_links:
            linked_marker = self.markers.get(link_id)
            if linked_marker:
                linked_markers.append(linked_marker)
        
        return linked_markers
    
    def _detect_modality(self, file_path: str) -> ModalityType:
        """Detect modality from file extension"""
        for modality, processor in self.processors.items():
            if processor.is_supported(file_path):
                return modality
        return ModalityType.TEXT
    
    def _generate_marker_id(self, file_path: str) -> str:
        """Generate unique marker ID"""
        # Use file path and timestamp for uniqueness
        content = f"{file_path}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _create_media_metadata(self, file_path: str) -> MediaMetadata:
        """Create metadata for media file"""
        stat = os.stat(file_path)
        
        # Calculate checksum
        with open(file_path, 'rb') as f:
            content = f.read()
            checksum = hashlib.sha256(content).hexdigest()
        
        return MediaMetadata(
            file_path=file_path,
            file_size=stat.st_size,
            mime_type=self._get_mime_type(file_path),
            checksum=checksum
        )
    
    def _get_mime_type(self, file_path: str) -> str:
        """Get MIME type from file extension"""
        ext = Path(file_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
            '.gif': 'image/gif', '.bmp': 'image/bmp', '.tiff': 'image/tiff',
            '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.flac': 'audio/flac',
            '.ogg': 'audio/ogg', '.m4a': 'audio/mp4', '.aac': 'audio/aac',
            '.mp4': 'video/mp4', '.avi': 'video/x-msvideo', '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska', '.wmv': 'video/x-ms-wmv', '.webm': 'video/webm'
        }
        return mime_types.get(ext, 'application/octet-stream')
    
    def _update_cross_modal_index(self, marker: MultimodalMarker):
        """Update cross-modal search index"""
        for modality, features in marker.features.items():
            for keyword in features.keywords:
                if keyword not in self.cross_modal_index:
                    self.cross_modal_index[keyword] = []
                if marker.id not in self.cross_modal_index[keyword]:
                    self.cross_modal_index[keyword].append(marker.id)
    
    def _save_marker(self, marker: MultimodalMarker):
        """Save marker to disk"""
        try:
            marker_file = self.storage_path / f"{marker.id}.json"
            
            # Convert marker to serializable format
            marker_data = {
                'id': marker.id,
                'primary_content': marker.primary_content,
                'modalities': {k.value: v for k, v in marker.modalities.items()},
                'features': {
                    k.value: {
                        'modality': v.modality.value,
                        'features': v.features,
                        'metadata': v.metadata,
                        'confidence': v.confidence,
                        'description': v.description,
                        'keywords': v.keywords,
                        'embeddings': v.embeddings
                    } for k, v in marker.features.items()
                },
                'cross_modal_links': marker.cross_modal_links,
                'metadata': {
                    'file_path': marker.metadata.file_path,
                    'file_size': marker.metadata.file_size,
                    'mime_type': marker.metadata.mime_type,
                    'duration': marker.metadata.duration,
                    'dimensions': marker.metadata.dimensions,
                    'sample_rate': marker.metadata.sample_rate,
                    'fps': marker.metadata.fps,
                    'created_at': marker.metadata.created_at.isoformat(),
                    'checksum': marker.metadata.checksum
                } if marker.metadata else None,
                'created_at': marker.created_at.isoformat(),
                'usage_count': marker.usage_count,
                'reputation': marker.reputation
            }
            
            with open(marker_file, 'w') as f:
                json.dump(marker_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save marker {marker.id}: {e}")
    
    def _load_markers(self):
        """Load markers from disk"""
        try:
            for marker_file in self.storage_path.glob("*.json"):
                with open(marker_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct marker
                marker = self._reconstruct_marker(data)
                if marker:
                    self.markers[marker.id] = marker
                    self._update_cross_modal_index(marker)
            
            logger.info(f"Loaded {len(self.markers)} multimodal markers")
            
        except Exception as e:
            logger.error(f"Failed to load markers: {e}")
    
    def _reconstruct_marker(self, data: Dict[str, Any]) -> Optional[MultimodalMarker]:
        """Reconstruct marker from saved data"""
        try:
            # Reconstruct modalities
            modalities = {ModalityType(k): v for k, v in data['modalities'].items()}
            
            # Reconstruct features
            features = {}
            for k, v in data['features'].items():
                features[ModalityType(k)] = ModalityFeatures(
                    modality=ModalityType(v['modality']),
                    features=v['features'],
                    metadata=v['metadata'],
                    confidence=v['confidence'],
                    description=v.get('description'),
                    keywords=v.get('keywords', []),
                    embeddings=v.get('embeddings')
                )
            
            # Reconstruct metadata
            metadata = None
            if data.get('metadata'):
                md = data['metadata']
                metadata = MediaMetadata(
                    file_path=md['file_path'],
                    file_size=md['file_size'],
                    mime_type=md['mime_type'],
                    duration=md.get('duration'),
                    dimensions=tuple(md['dimensions']) if md.get('dimensions') else None,
                    sample_rate=md.get('sample_rate'),
                    fps=md.get('fps'),
                    created_at=datetime.fromisoformat(md['created_at']),
                    checksum=md.get('checksum')
                )
            
            return MultimodalMarker(
                id=data['id'],
                primary_content=data['primary_content'],
                modalities=modalities,
                features=features,
                cross_modal_links=data.get('cross_modal_links', []),
                metadata=metadata,
                created_at=datetime.fromisoformat(data['created_at']),
                usage_count=data.get('usage_count', 0),
                reputation=data.get('reputation', 0.0)
            )
            
        except Exception as e:
            logger.error(f"Failed to reconstruct marker: {e}")
            return None

# Default configuration
DEFAULT_MULTIMODAL_CONFIG = {
    'storage_path': 'multimodal_storage',
    'image_config': {
        'clip_model': 'openai/clip-vit-base-patch32',
        'blip_model': 'Salesforce/blip-image-captioning-base'
    },
    'audio_config': {
        'sample_rate': 22050,
        'max_duration': 30
    },
    'video_config': {
        'max_frames': 30,
        'image_config': {},
        'audio_config': {}
    }
}

def create_multimodal_manager(config: Optional[Dict[str, Any]] = None) -> MultimodalMemoryManager:
    """Create multimodal memory manager with default or custom config"""
    if config is None:
        config = DEFAULT_MULTIMODAL_CONFIG
    
    return MultimodalMemoryManager(config)