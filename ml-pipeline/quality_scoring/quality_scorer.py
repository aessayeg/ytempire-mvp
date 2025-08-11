#!/usr/bin/env python3
"""
AI/ML Content Quality Scoring System for YTEmpire
Advanced multi-modal quality assessment using computer vision, NLP, and audio analysis
"""

import os
import json
import logging
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import librosa
import whisper
from transformers import (
    BertTokenizer, BertModel,
    CLIPProcessor, CLIPModel,
    pipeline
)
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import aiofiles
from pathlib import Path
import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VideoAnalysisMetrics:
    """Comprehensive video analysis metrics"""
    # Visual Quality Metrics
    resolution_score: float = 0.0
    sharpness_score: float = 0.0
    brightness_score: float = 0.0
    contrast_score: float = 0.0
    color_balance_score: float = 0.0
    visual_stability_score: float = 0.0
    
    # Audio Quality Metrics
    audio_clarity_score: float = 0.0
    volume_consistency_score: float = 0.0
    noise_level_score: float = 0.0
    speech_quality_score: float = 0.0
    
    # Content Quality Metrics
    script_coherence_score: float = 0.0
    topic_relevance_score: float = 0.0
    engagement_potential_score: float = 0.0
    information_density_score: float = 0.0
    
    # Technical Metrics
    encoding_quality_score: float = 0.0
    file_size_efficiency_score: float = 0.0
    duration_appropriateness_score: float = 0.0
    
    # Composite Scores
    overall_quality_score: float = 0.0
    predicted_engagement_score: float = 0.0
    monetization_potential_score: float = 0.0
    
    # Metadata
    analysis_timestamp: str = ""
    processing_time_seconds: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)

@dataclass
class QualityScoringConfig:
    """Configuration for quality scoring system"""
    
    # Model paths and configurations
    whisper_model_size: str = "base"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    bert_model_name: str = "bert-base-uncased"
    
    # Quality thresholds
    min_acceptable_score: float = 0.65
    excellent_score_threshold: float = 0.85
    
    # Processing settings
    max_concurrent_analyses: int = 4
    frame_sample_rate: int = 30  # Analyze every 30th frame
    audio_chunk_duration: float = 10.0  # seconds
    
    # Feature weights for composite scoring
    visual_weight: float = 0.25
    audio_weight: float = 0.25
    content_weight: float = 0.30
    technical_weight: float = 0.20
    
    # Performance targets
    target_processing_time: float = 120.0  # seconds per video
    max_video_duration: float = 1200.0  # 20 minutes
    
    # Storage and caching
    cache_results: bool = True
    cache_duration_hours: int = 24
    database_path: str = "quality_scores.db"

class VisualQualityAnalyzer:
    """Analyzes visual quality of video frames"""
    
    def __init__(self, config: QualityScoringConfig):
        self.config = config
        self.clip_processor = CLIPProcessor.from_pretrained(config.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(config.clip_model_name)
        
        # Visual quality detection models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model.to(self.device)
        
    async def analyze_video_quality(self, video_path: str) -> Dict[str, float]:
        """Analyze visual quality metrics of video"""
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Sample frames for analysis
            sample_indices = np.linspace(0, total_frames-1, 
                                       min(total_frames//self.config.frame_sample_rate, 100), 
                                       dtype=int)
            
            frames_metrics = []
            
            for frame_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Analyze frame quality
                frame_metrics = await self._analyze_frame(frame)
                frames_metrics.append(frame_metrics)
            
            cap.release()
            
            # Aggregate frame metrics
            aggregated_metrics = self._aggregate_frame_metrics(frames_metrics)
            
            # Add resolution score
            resolution_score = self._calculate_resolution_score(width, height)
            aggregated_metrics['resolution_score'] = resolution_score
            
            return aggregated_metrics
            
        except Exception as e:
            logger.error(f"Visual analysis error: {str(e)}")
            return self._get_default_visual_metrics()
    
    async def _analyze_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """Analyze individual frame quality"""
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Calculate sharpness (Laplacian variance)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize
        
        # Calculate brightness
        brightness = np.mean(gray_frame) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Optimal around 0.5
        
        # Calculate contrast (standard deviation)
        contrast = np.std(gray_frame) / 255.0
        contrast_score = min(contrast * 4, 1.0)  # Normalize
        
        # Color balance analysis
        b, g, r = cv2.split(frame)
        color_balance_score = 1.0 - (np.std([np.mean(r), np.mean(g), np.mean(b)]) / 255.0)
        
        return {
            'sharpness_score': sharpness_score,
            'brightness_score': brightness_score,
            'contrast_score': contrast_score,
            'color_balance_score': color_balance_score
        }
    
    def _aggregate_frame_metrics(self, frames_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across all analyzed frames"""
        
        if not frames_metrics:
            return self._get_default_visual_metrics()
        
        # Calculate means and stability
        metrics_arrays = {}
        for key in frames_metrics[0].keys():
            values = [frame[key] for frame in frames_metrics]
            metrics_arrays[key] = np.array(values)
        
        aggregated = {}
        for key, values in metrics_arrays.items():
            aggregated[key] = float(np.mean(values))
        
        # Calculate visual stability (low variance = high stability)
        stability_scores = []
        for key, values in metrics_arrays.items():
            if len(values) > 1:
                stability = 1.0 - min(np.std(values), 0.5) * 2
                stability_scores.append(stability)
        
        aggregated['visual_stability_score'] = float(np.mean(stability_scores)) if stability_scores else 0.8
        
        return aggregated
    
    def _calculate_resolution_score(self, width: int, height: int) -> float:
        """Calculate resolution quality score"""
        
        total_pixels = width * height
        
        # Resolution tiers
        if total_pixels >= 3840 * 2160:  # 4K
            return 1.0
        elif total_pixels >= 1920 * 1080:  # 1080p
            return 0.9
        elif total_pixels >= 1280 * 720:   # 720p
            return 0.7
        elif total_pixels >= 854 * 480:    # 480p
            return 0.5
        else:
            return 0.3
    
    def _get_default_visual_metrics(self) -> Dict[str, float]:
        """Default visual metrics when analysis fails"""
        return {
            'resolution_score': 0.5,
            'sharpness_score': 0.5,
            'brightness_score': 0.5,
            'contrast_score': 0.5,
            'color_balance_score': 0.5,
            'visual_stability_score': 0.5
        }

class AudioQualityAnalyzer:
    """Analyzes audio quality and speech content"""
    
    def __init__(self, config: QualityScoringConfig):
        self.config = config
        self.whisper_model = whisper.load_model(config.whisper_model_size)
        
    async def analyze_audio_quality(self, video_path: str) -> Dict[str, float]:
        """Analyze audio quality metrics"""
        
        try:
            # Extract audio from video
            audio_path = await self._extract_audio(video_path)
            
            # Load audio data
            audio_data, sr = librosa.load(audio_path, sr=None)
            
            # Analyze audio quality
            audio_metrics = await self._analyze_audio_properties(audio_data, sr)
            
            # Analyze speech quality
            speech_metrics = await self._analyze_speech_quality(audio_path)
            
            # Combine metrics
            combined_metrics = {**audio_metrics, **speech_metrics}
            
            # Cleanup temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return combined_metrics
            
        except Exception as e:
            logger.error(f"Audio analysis error: {str(e)}")
            return self._get_default_audio_metrics()
    
    async def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video file"""
        
        audio_path = f"{video_path}_temp_audio.wav"
        
        # Use ffmpeg to extract audio
        cmd = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}" -y'
        
        process = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        
        await process.communicate()
        
        if not os.path.exists(audio_path):
            raise RuntimeError("Failed to extract audio from video")
        
        return audio_path
    
    async def _analyze_audio_properties(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze basic audio properties"""
        
        # Audio clarity (spectral centroid)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        clarity_score = min(np.mean(spectral_centroids) / 2000.0, 1.0)
        
        # Volume consistency (RMS energy)
        rms_energy = librosa.feature.rms(y=audio_data)[0]
        volume_std = np.std(rms_energy)
        volume_consistency_score = max(0.0, 1.0 - volume_std * 10)
        
        # Noise level analysis (using zero crossing rate)
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        noise_level = np.mean(zcr)
        noise_level_score = max(0.0, 1.0 - noise_level * 2)
        
        return {
            'audio_clarity_score': float(clarity_score),
            'volume_consistency_score': float(volume_consistency_score),
            'noise_level_score': float(noise_level_score)
        }
    
    async def _analyze_speech_quality(self, audio_path: str) -> Dict[str, float]:
        """Analyze speech quality using Whisper"""
        
        try:
            # Transcribe audio
            result = self.whisper_model.transcribe(audio_path)
            
            transcript = result["text"]
            segments = result.get("segments", [])
            
            if not transcript.strip():
                return {'speech_quality_score': 0.0}
            
            # Calculate speech quality based on confidence and completeness
            if segments:
                # Average confidence from segments
                avg_confidence = np.mean([segment.get("confidence", 0.5) for segment in segments])
                
                # Word density (words per minute)
                total_duration = segments[-1]["end"] if segments else 1.0
                word_count = len(transcript.split())
                words_per_minute = (word_count / total_duration) * 60
                
                # Optimal WPM is around 150-160
                wpm_score = 1.0 - abs(words_per_minute - 155) / 155.0
                wpm_score = max(0.0, min(1.0, wpm_score))
                
                speech_quality_score = (avg_confidence * 0.7) + (wpm_score * 0.3)
            else:
                speech_quality_score = 0.5
            
            return {'speech_quality_score': float(speech_quality_score)}
            
        except Exception as e:
            logger.warning(f"Speech analysis failed: {str(e)}")
            return {'speech_quality_score': 0.5}
    
    def _get_default_audio_metrics(self) -> Dict[str, float]:
        """Default audio metrics when analysis fails"""
        return {
            'audio_clarity_score': 0.5,
            'volume_consistency_score': 0.5,
            'noise_level_score': 0.5,
            'speech_quality_score': 0.5
        }

class ContentQualityAnalyzer:
    """Analyzes content quality using NLP and semantic analysis"""
    
    def __init__(self, config: QualityScoringConfig):
        self.config = config
        
        # Initialize models
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
        self.bert_model = BertModel.from_pretrained(config.bert_model_name)
        
        # Sentiment analysis pipeline
        self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                         model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        
        # Topic classification pipeline
        self.topic_classifier = pipeline("zero-shot-classification",
                                        model="facebook/bart-large-mnli")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model.to(self.device)
    
    async def analyze_content_quality(self, script_text: str, target_topic: str = "") -> Dict[str, float]:
        """Analyze content quality metrics"""
        
        try:
            if not script_text.strip():
                return self._get_default_content_metrics()
            
            # Analyze script coherence
            coherence_score = await self._analyze_script_coherence(script_text)
            
            # Analyze topic relevance
            relevance_score = await self._analyze_topic_relevance(script_text, target_topic)
            
            # Analyze engagement potential
            engagement_score = await self._analyze_engagement_potential(script_text)
            
            # Analyze information density
            info_density_score = await self._analyze_information_density(script_text)
            
            return {
                'script_coherence_score': coherence_score,
                'topic_relevance_score': relevance_score,
                'engagement_potential_score': engagement_score,
                'information_density_score': info_density_score
            }
            
        except Exception as e:
            logger.error(f"Content analysis error: {str(e)}")
            return self._get_default_content_metrics()
    
    async def _analyze_script_coherence(self, text: str) -> float:
        """Analyze coherence and flow of script"""
        
        sentences = text.split('. ')
        if len(sentences) < 2:
            return 0.5
        
        # Encode sentences
        sentence_embeddings = []
        
        for sentence in sentences[:20]:  # Limit for performance
            if sentence.strip():
                inputs = self.tokenizer(sentence, return_tensors='pt', 
                                      truncation=True, padding=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1)
                    sentence_embeddings.append(embedding.cpu().numpy())
        
        if len(sentence_embeddings) < 2:
            return 0.5
        
        # Calculate semantic similarity between adjacent sentences
        similarities = []
        for i in range(len(sentence_embeddings) - 1):
            sim = np.dot(sentence_embeddings[i].flatten(), 
                        sentence_embeddings[i+1].flatten())
            sim = sim / (np.linalg.norm(sentence_embeddings[i]) * 
                        np.linalg.norm(sentence_embeddings[i+1]))
            similarities.append(sim)
        
        # Higher similarity indicates better coherence
        coherence_score = float(np.mean(similarities))
        return max(0.0, min(1.0, (coherence_score + 1) / 2))  # Normalize to 0-1
    
    async def _analyze_topic_relevance(self, text: str, target_topic: str) -> float:
        """Analyze relevance to target topic"""
        
        if not target_topic:
            return 0.8  # Default score when no target topic
        
        try:
            # Use zero-shot classification
            candidate_labels = [target_topic, "general", "off-topic"]
            result = self.topic_classifier(text[:1000], candidate_labels)  # Limit text length
            
            # Find relevance to target topic
            for label, score in zip(result['labels'], result['scores']):
                if label.lower() == target_topic.lower():
                    return float(score)
            
            return 0.5  # Default if target topic not found
            
        except Exception as e:
            logger.warning(f"Topic relevance analysis failed: {str(e)}")
            return 0.5
    
    async def _analyze_engagement_potential(self, text: str) -> float:
        """Analyze potential for audience engagement"""
        
        # Sentiment analysis
        sentiment_result = self.sentiment_analyzer(text[:1000])
        sentiment_score = sentiment_result[0]['score']
        
        # Positive sentiment generally has higher engagement
        if sentiment_result[0]['label'] == 'NEGATIVE':
            sentiment_score = 1 - sentiment_score
        
        # Question count (questions engage audience)
        question_count = text.count('?')
        question_score = min(question_count * 0.1, 0.3)
        
        # Exclamation points (indicate enthusiasm)
        exclamation_count = text.count('!')
        enthusiasm_score = min(exclamation_count * 0.05, 0.2)
        
        # Word variety (lexical diversity)
        words = text.lower().split()
        unique_words = set(words)
        diversity_score = len(unique_words) / max(len(words), 1) if words else 0
        
        # Combine scores
        engagement_score = (sentiment_score * 0.4 + 
                          question_score + 
                          enthusiasm_score + 
                          diversity_score * 0.4)
        
        return float(min(engagement_score, 1.0))
    
    async def _analyze_information_density(self, text: str) -> float:
        """Analyze information content density"""
        
        words = text.split()
        if not words:
            return 0.0
        
        # Count informative words (nouns, verbs, adjectives)
        import nltk
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            pass
        
        try:
            from nltk import pos_tag, word_tokenize
            
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            informative_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']
            informative_count = sum(1 for word, tag in pos_tags if tag in informative_tags)
            
            density_score = informative_count / len(tokens)
            return float(min(density_score * 2, 1.0))  # Normalize
            
        except Exception:
            # Fallback: simple word length analysis
            avg_word_length = np.mean([len(word) for word in words])
            return float(min(avg_word_length / 8.0, 1.0))
    
    def _get_default_content_metrics(self) -> Dict[str, float]:
        """Default content metrics when analysis fails"""
        return {
            'script_coherence_score': 0.5,
            'topic_relevance_score': 0.5,
            'engagement_potential_score': 0.5,
            'information_density_score': 0.5
        }

class TechnicalQualityAnalyzer:
    """Analyzes technical aspects of video"""
    
    def __init__(self, config: QualityScoringConfig):
        self.config = config
    
    async def analyze_technical_quality(self, video_path: str) -> Dict[str, float]:
        """Analyze technical quality metrics"""
        
        try:
            # Get video properties
            properties = await self._get_video_properties(video_path)
            
            # Calculate technical scores
            encoding_score = self._analyze_encoding_quality(properties)
            size_efficiency_score = self._analyze_file_size_efficiency(properties)
            duration_score = self._analyze_duration_appropriateness(properties)
            
            return {
                'encoding_quality_score': encoding_score,
                'file_size_efficiency_score': size_efficiency_score,
                'duration_appropriateness_score': duration_score
            }
            
        except Exception as e:
            logger.error(f"Technical analysis error: {str(e)}")
            return self._get_default_technical_metrics()
    
    async def _get_video_properties(self, video_path: str) -> Dict[str, Any]:
        """Extract video properties using ffprobe"""
        
        cmd = f'ffprobe -v quiet -print_format json -show_format -show_streams "{video_path}"'
        
        process = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {stderr.decode()}")
        
        return json.loads(stdout.decode())
    
    def _analyze_encoding_quality(self, properties: Dict[str, Any]) -> float:
        """Analyze encoding quality"""
        
        try:
            video_stream = None
            for stream in properties['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                return 0.5
            
            # Check codec
            codec = video_stream.get('codec_name', '').lower()
            codec_score = 1.0 if codec in ['h264', 'h265', 'vp9', 'av1'] else 0.6
            
            # Check bitrate
            bitrate = int(video_stream.get('bit_rate', 0))
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            
            if width and height:
                pixels_per_second = width * height * float(video_stream.get('r_frame_rate', '30/1').split('/')[0])
                target_bpp = 0.1  # bits per pixel target
                target_bitrate = pixels_per_second * target_bpp
                
                bitrate_ratio = bitrate / max(target_bitrate, 1)
                bitrate_score = min(bitrate_ratio, 2.0) / 2.0  # Normalize
            else:
                bitrate_score = 0.5
            
            return float((codec_score + bitrate_score) / 2)
            
        except Exception:
            return 0.5
    
    def _analyze_file_size_efficiency(self, properties: Dict[str, Any]) -> float:
        """Analyze file size efficiency"""
        
        try:
            format_info = properties.get('format', {})
            file_size = int(format_info.get('size', 0))
            duration = float(format_info.get('duration', 0))
            
            if not file_size or not duration:
                return 0.5
            
            # Calculate size per minute (MB/min)
            size_per_minute = (file_size / (1024 * 1024)) / (duration / 60)
            
            # Optimal range: 10-50 MB/min for good quality
            if 10 <= size_per_minute <= 50:
                efficiency_score = 1.0
            elif size_per_minute < 10:
                efficiency_score = size_per_minute / 10
            else:
                efficiency_score = max(0.2, 1.0 - ((size_per_minute - 50) / 100))
            
            return float(efficiency_score)
            
        except Exception:
            return 0.5
    
    def _analyze_duration_appropriateness(self, properties: Dict[str, Any]) -> float:
        """Analyze if duration is appropriate for content type"""
        
        try:
            duration = float(properties['format'].get('duration', 0))
            
            # YouTube optimal ranges
            if 60 <= duration <= 600:  # 1-10 minutes
                return 1.0
            elif 30 <= duration <= 1200:  # 30 seconds - 20 minutes
                return 0.8
            elif duration >= 1200:  # > 20 minutes
                return max(0.4, 1.0 - ((duration - 1200) / 3600))
            else:  # < 30 seconds
                return duration / 30
            
        except Exception:
            return 0.5
    
    def _get_default_technical_metrics(self) -> Dict[str, float]:
        """Default technical metrics when analysis fails"""
        return {
            'encoding_quality_score': 0.5,
            'file_size_efficiency_score': 0.5,
            'duration_appropriateness_score': 0.5
        }

class QualityScoreDatabase:
    """Database for storing and retrieving quality scores"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_hash TEXT UNIQUE NOT NULL,
                    video_path TEXT NOT NULL,
                    analysis_timestamp TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    processing_time REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_video_hash ON quality_scores(video_hash)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_overall_score ON quality_scores(overall_score)
            """)
    
    def get_cached_score(self, video_hash: str, max_age_hours: int = 24) -> Optional[VideoAnalysisMetrics]:
        """Get cached quality score if available and recent"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT metrics_json FROM quality_scores 
                WHERE video_hash = ? 
                AND datetime(created_at) > datetime('now', '-{} hours')
                ORDER BY created_at DESC LIMIT 1
            """.format(max_age_hours), (video_hash,))
            
            row = cursor.fetchone()
            if row:
                metrics_dict = json.loads(row[0])
                return VideoAnalysisMetrics(**metrics_dict)
        
        return None
    
    def save_score(self, video_hash: str, video_path: str, metrics: VideoAnalysisMetrics):
        """Save quality score to database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO quality_scores 
                (video_hash, video_path, analysis_timestamp, metrics_json, overall_score, processing_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                video_hash,
                video_path,
                metrics.analysis_timestamp,
                json.dumps(asdict(metrics)),
                metrics.overall_quality_score,
                metrics.processing_time_seconds
            ))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get quality scoring statistics"""
        
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Count total analyses
            cursor = conn.execute("SELECT COUNT(*) FROM quality_scores")
            stats['total_analyses'] = cursor.fetchone()[0]
            
            # Average scores
            cursor = conn.execute("""
                SELECT AVG(overall_score) as avg_score,
                       MIN(overall_score) as min_score,
                       MAX(overall_score) as max_score
                FROM quality_scores
            """)
            row = cursor.fetchone()
            stats.update({
                'average_score': row[0] or 0.0,
                'min_score': row[1] or 0.0,
                'max_score': row[2] or 0.0
            })
            
            # Recent analyses (last 24 hours)
            cursor = conn.execute("""
                SELECT COUNT(*) FROM quality_scores 
                WHERE datetime(created_at) > datetime('now', '-24 hours')
            """)
            stats['recent_analyses'] = cursor.fetchone()[0]
            
            return stats

class ContentQualityScorer:
    """Main quality scoring orchestrator"""
    
    def __init__(self, config: QualityScoringConfig):
        self.config = config
        
        # Initialize analyzers
        self.visual_analyzer = VisualQualityAnalyzer(config)
        self.audio_analyzer = AudioQualityAnalyzer(config)
        self.content_analyzer = ContentQualityAnalyzer(config)
        self.technical_analyzer = TechnicalQualityAnalyzer(config)
        
        # Initialize database
        self.database = QualityScoreDatabase(config.database_path)
        
        # Thread pool for CPU-bound tasks
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_analyses)
    
    def _calculate_video_hash(self, video_path: str) -> str:
        """Calculate hash of video file for caching"""
        
        hasher = hashlib.md5()
        
        # Include file path and modification time
        hasher.update(video_path.encode())
        
        if os.path.exists(video_path):
            mtime = os.path.getmtime(video_path)
            hasher.update(str(mtime).encode())
        
        return hasher.hexdigest()
    
    async def score_video(self, 
                         video_path: str, 
                         script_text: str = "",
                         target_topic: str = "",
                         use_cache: bool = True) -> VideoAnalysisMetrics:
        """Comprehensive video quality scoring"""
        
        start_time = datetime.now()
        logger.info(f"Starting quality analysis for: {video_path}")
        
        # Check cache first
        video_hash = self._calculate_video_hash(video_path)
        
        if use_cache and self.config.cache_results:
            cached_metrics = self.database.get_cached_score(
                video_hash, self.config.cache_duration_hours
            )
            if cached_metrics:
                logger.info("Using cached quality score")
                return cached_metrics
        
        # Verify video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            # Run analyses in parallel
            visual_task = asyncio.create_task(self.visual_analyzer.analyze_video_quality(video_path))
            audio_task = asyncio.create_task(self.audio_analyzer.analyze_audio_quality(video_path))
            content_task = asyncio.create_task(self.content_analyzer.analyze_content_quality(script_text, target_topic))
            technical_task = asyncio.create_task(self.technical_analyzer.analyze_technical_quality(video_path))
            
            # Await all results
            visual_metrics = await visual_task
            audio_metrics = await audio_task
            content_metrics = await content_task
            technical_metrics = await technical_task
            
            # Create comprehensive metrics
            metrics = VideoAnalysisMetrics(
                # Visual metrics
                resolution_score=visual_metrics['resolution_score'],
                sharpness_score=visual_metrics['sharpness_score'],
                brightness_score=visual_metrics['brightness_score'],
                contrast_score=visual_metrics['contrast_score'],
                color_balance_score=visual_metrics['color_balance_score'],
                visual_stability_score=visual_metrics['visual_stability_score'],
                
                # Audio metrics
                audio_clarity_score=audio_metrics['audio_clarity_score'],
                volume_consistency_score=audio_metrics['volume_consistency_score'],
                noise_level_score=audio_metrics['noise_level_score'],
                speech_quality_score=audio_metrics['speech_quality_score'],
                
                # Content metrics
                script_coherence_score=content_metrics['script_coherence_score'],
                topic_relevance_score=content_metrics['topic_relevance_score'],
                engagement_potential_score=content_metrics['engagement_potential_score'],
                information_density_score=content_metrics['information_density_score'],
                
                # Technical metrics
                encoding_quality_score=technical_metrics['encoding_quality_score'],
                file_size_efficiency_score=technical_metrics['file_size_efficiency_score'],
                duration_appropriateness_score=technical_metrics['duration_appropriateness_score'],
                
                # Timestamp and processing time
                analysis_timestamp=datetime.now().isoformat(),
                processing_time_seconds=(datetime.now() - start_time).total_seconds()
            )
            
            # Calculate composite scores
            metrics = self._calculate_composite_scores(metrics)
            
            # Save to database
            if self.config.cache_results:
                self.database.save_score(video_hash, video_path, metrics)
            
            logger.info(f"Quality analysis completed. Overall score: {metrics.overall_quality_score:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Quality scoring failed: {str(e)}")
            raise
    
    def _calculate_composite_scores(self, metrics: VideoAnalysisMetrics) -> VideoAnalysisMetrics:
        """Calculate composite quality scores"""
        
        # Visual composite
        visual_score = np.mean([
            metrics.resolution_score,
            metrics.sharpness_score,
            metrics.brightness_score,
            metrics.contrast_score,
            metrics.color_balance_score,
            metrics.visual_stability_score
        ])
        
        # Audio composite
        audio_score = np.mean([
            metrics.audio_clarity_score,
            metrics.volume_consistency_score,
            metrics.noise_level_score,
            metrics.speech_quality_score
        ])
        
        # Content composite
        content_score = np.mean([
            metrics.script_coherence_score,
            metrics.topic_relevance_score,
            metrics.engagement_potential_score,
            metrics.information_density_score
        ])
        
        # Technical composite
        technical_score = np.mean([
            metrics.encoding_quality_score,
            metrics.file_size_efficiency_score,
            metrics.duration_appropriateness_score
        ])
        
        # Overall weighted score
        overall_score = (
            visual_score * self.config.visual_weight +
            audio_score * self.config.audio_weight +
            content_score * self.config.content_weight +
            technical_score * self.config.technical_weight
        )
        
        # Engagement prediction (machine learning would improve this)
        engagement_score = (content_score * 0.5 + visual_score * 0.3 + audio_score * 0.2)
        
        # Monetization potential
        monetization_score = min(overall_score * 1.1, 1.0)  # Slightly favor high overall quality
        
        # Calculate confidence interval (simplified)
        score_variance = np.var([visual_score, audio_score, content_score, technical_score])
        confidence_range = np.sqrt(score_variance) * 0.1
        confidence_interval = (
            max(0.0, overall_score - confidence_range),
            min(1.0, overall_score + confidence_range)
        )
        
        # Update metrics
        metrics.overall_quality_score = float(overall_score)
        metrics.predicted_engagement_score = float(engagement_score)
        metrics.monetization_potential_score = float(monetization_score)
        metrics.confidence_interval = confidence_interval
        
        return metrics
    
    async def batch_score_videos(self, video_paths: List[str], **kwargs) -> List[VideoAnalysisMetrics]:
        """Score multiple videos in batch"""
        
        semaphore = asyncio.Semaphore(self.config.max_concurrent_analyses)
        
        async def score_with_semaphore(video_path: str):
            async with semaphore:
                return await self.score_video(video_path, **kwargs)
        
        tasks = [score_with_semaphore(path) for path in video_paths]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_quality_report(self, metrics: VideoAnalysisMetrics) -> Dict[str, Any]:
        """Generate human-readable quality report"""
        
        def get_grade(score: float) -> str:
            if score >= 0.9: return "A+"
            elif score >= 0.8: return "A"
            elif score >= 0.7: return "B"
            elif score >= 0.6: return "C"
            elif score >= 0.5: return "D"
            else: return "F"
        
        def get_recommendation(score: float) -> str:
            if score >= self.config.excellent_score_threshold:
                return "Excellent quality - ready for publication"
            elif score >= self.config.min_acceptable_score:
                return "Good quality - minor improvements possible"
            else:
                return "Quality needs improvement before publication"
        
        report = {
            "overall_assessment": {
                "score": metrics.overall_quality_score,
                "grade": get_grade(metrics.overall_quality_score),
                "recommendation": get_recommendation(metrics.overall_quality_score),
                "confidence_interval": metrics.confidence_interval
            },
            "category_scores": {
                "visual_quality": {
                    "resolution": {"score": metrics.resolution_score, "grade": get_grade(metrics.resolution_score)},
                    "sharpness": {"score": metrics.sharpness_score, "grade": get_grade(metrics.sharpness_score)},
                    "brightness": {"score": metrics.brightness_score, "grade": get_grade(metrics.brightness_score)},
                    "contrast": {"score": metrics.contrast_score, "grade": get_grade(metrics.contrast_score)},
                    "color_balance": {"score": metrics.color_balance_score, "grade": get_grade(metrics.color_balance_score)},
                    "stability": {"score": metrics.visual_stability_score, "grade": get_grade(metrics.visual_stability_score)}
                },
                "audio_quality": {
                    "clarity": {"score": metrics.audio_clarity_score, "grade": get_grade(metrics.audio_clarity_score)},
                    "volume_consistency": {"score": metrics.volume_consistency_score, "grade": get_grade(metrics.volume_consistency_score)},
                    "noise_level": {"score": metrics.noise_level_score, "grade": get_grade(metrics.noise_level_score)},
                    "speech_quality": {"score": metrics.speech_quality_score, "grade": get_grade(metrics.speech_quality_score)}
                },
                "content_quality": {
                    "script_coherence": {"score": metrics.script_coherence_score, "grade": get_grade(metrics.script_coherence_score)},
                    "topic_relevance": {"score": metrics.topic_relevance_score, "grade": get_grade(metrics.topic_relevance_score)},
                    "engagement_potential": {"score": metrics.engagement_potential_score, "grade": get_grade(metrics.engagement_potential_score)},
                    "information_density": {"score": metrics.information_density_score, "grade": get_grade(metrics.information_density_score)}
                },
                "technical_quality": {
                    "encoding_quality": {"score": metrics.encoding_quality_score, "grade": get_grade(metrics.encoding_quality_score)},
                    "file_size_efficiency": {"score": metrics.file_size_efficiency_score, "grade": get_grade(metrics.file_size_efficiency_score)},
                    "duration_appropriateness": {"score": metrics.duration_appropriateness_score, "grade": get_grade(metrics.duration_appropriateness_score)}
                }
            },
            "predictions": {
                "engagement_score": metrics.predicted_engagement_score,
                "monetization_potential": metrics.monetization_potential_score
            },
            "metadata": {
                "analysis_timestamp": metrics.analysis_timestamp,
                "processing_time": metrics.processing_time_seconds
            }
        }
        
        return report
    
    def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)

async def main():
    """Example usage of the quality scoring system"""
    
    # Configuration
    config = QualityScoringConfig(
        min_acceptable_score=0.65,
        excellent_score_threshold=0.85,
        max_concurrent_analyses=2
    )
    
    # Initialize scorer
    scorer = ContentQualityScorer(config)
    
    try:
        # Example video analysis
        video_path = "test_videos/sample.mp4"
        script_text = "Welcome to our channel! Today we'll explore the fascinating world of AI..."
        target_topic = "artificial intelligence"
        
        if os.path.exists(video_path):
            # Score the video
            metrics = await scorer.score_video(
                video_path=video_path,
                script_text=script_text,
                target_topic=target_topic
            )
            
            # Generate report
            report = scorer.get_quality_report(metrics)
            
            print("Quality Analysis Report:")
            print("="*50)
            print(f"Overall Score: {report['overall_assessment']['score']:.3f} ({report['overall_assessment']['grade']})")
            print(f"Recommendation: {report['overall_assessment']['recommendation']}")
            print(f"Processing Time: {metrics.processing_time_seconds:.2f}s")
            
            # Print category scores
            for category, scores in report['category_scores'].items():
                print(f"\n{category.replace('_', ' ').title()}:")
                for metric, data in scores.items():
                    print(f"  {metric.replace('_', ' ').title()}: {data['score']:.3f} ({data['grade']})")
        else:
            print("Test video not found. Quality scoring system is ready.")
        
    finally:
        scorer.close()

if __name__ == "__main__":
    asyncio.run(main())