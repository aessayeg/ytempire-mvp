"""
Stock Footage Service
Integrates with Pexels and Pixabay APIs for free stock footage
Based on YTEMPIRE Video Generation Architecture documentation
"""
import os
import asyncio
import aiohttp
import logging
import random
from typing import Dict, List, Optional, Any
from pathlib import Path
import hashlib
import json
from datetime import datetime, timedelta
import aiofiles

logger = logging.getLogger(__name__)


class StockFootageService:
    """
    Primary method for news, commentary, and general content
    Handles 60% of video generation according to architecture spec
    """
    
    def __init__(self):
        self.pexels_api_key = os.getenv("PEXELS_API_KEY", "")
        self.pixabay_api_key = os.getenv("PIXABAY_API_KEY", "")
        
        # Local cache directory for footage
        self.cache_dir = Path("./storage/footage_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata file
        self.cache_metadata_file = self.cache_dir / "metadata.json"
        self.cache_metadata = self._load_cache_metadata()
        
        # API endpoints
        self.pexels_base_url = "https://api.pexels.com/v1"
        self.pixabay_base_url = "https://pixabay.com/api"
        
        # Motion effects for Ken Burns effect
        self.motion_effects = [
            'zoom_in_slow',     # 1.0x to 1.2x over clip duration
            'zoom_out_slow',    # 1.2x to 1.0x
            'pan_left_right',   # Horizontal movement
            'pan_top_bottom',   # Vertical movement
            'rotate_subtle'     # -2 to +2 degrees
        ]
        
    def _load_cache_metadata(self) -> Dict:
        """Load cache metadata from disk"""
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache_metadata(self):
        """Save cache metadata to disk"""
        try:
            with open(self.cache_metadata_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    async def generate_visual_sequence(
        self,
        script_segments: List[Dict],
        content_type: str = "general"
    ) -> List[Dict]:
        """
        Maps script segments to relevant stock footage
        Returns visual timeline with footage clips
        """
        visual_timeline = []
        
        for segment in script_segments:
            # Extract keywords using simple NLP
            keywords = self.extract_visual_keywords(segment.get('text', ''))
            
            # Search for relevant footage (cached first, then API)
            footage_clips = await self.find_footage(keywords, segment.get('duration', 10))
            
            # Apply Ken Burns effect for dynamism
            processed_clips = self.apply_motion_effects(footage_clips)
            
            visual_timeline.append({
                'segment_id': segment.get('id', f"seg_{len(visual_timeline)}"),
                'clips': processed_clips,
                'duration': segment.get('duration', 10),
                'transition': 'crossfade',
                'text_overlay': segment.get('point', '')
            })
        
        return visual_timeline
    
    def extract_visual_keywords(self, text: str) -> List[str]:
        """Extract relevant visual keywords from text"""
        # Simple keyword extraction (in production, use NLP)
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for', 'with', 'by', 'from', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'over', 'up', 'down', 'out', 'off', 'on', 'in'}
        
        words = text.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Take top 5 keywords
        return keywords[:5] if keywords else ['abstract', 'background']
    
    async def find_footage(
        self,
        keywords: List[str],
        duration_needed: float = 10
    ) -> List[Dict]:
        """
        Find relevant footage from cache or APIs
        """
        query = ' '.join(keywords)
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        # Check cache first
        if cache_key in self.cache_metadata:
            cached_data = self.cache_metadata[cache_key]
            if self._is_cache_valid(cached_data):
                logger.info(f"Using cached footage for query: {query}")
                return cached_data['clips']
        
        # Search from APIs if not in cache
        clips = []
        
        # Try Pexels first
        if self.pexels_api_key:
            pexels_clips = await self._search_pexels(query)
            clips.extend(pexels_clips)
        
        # Then Pixabay
        if self.pixabay_api_key:
            pixabay_clips = await self._search_pixabay(query)
            clips.extend(pixabay_clips)
        
        # If no API keys or no results, use mock data
        if not clips:
            clips = self._generate_mock_footage(keywords, duration_needed)
        
        # Cache the results
        self.cache_metadata[cache_key] = {
            'clips': clips,
            'cached_at': datetime.utcnow().isoformat(),
            'query': query
        }
        self._save_cache_metadata()
        
        return clips
    
    async def _search_pexels(self, query: str) -> List[Dict]:
        """Search Pexels API for videos and images"""
        clips = []
        
        try:
            headers = {'Authorization': self.pexels_api_key}
            
            # Search videos first
            async with aiohttp.ClientSession() as session:
                # Search for videos
                video_url = f"{self.pexels_base_url}/videos/search"
                params = {'query': query, 'per_page': 5}
                
                async with session.get(video_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        for video in data.get('videos', [])[:3]:
                            clips.append({
                                'type': 'video',
                                'url': video['video_files'][0]['link'] if video.get('video_files') else '',
                                'duration': video.get('duration', 10),
                                'source': 'pexels',
                                'id': video['id']
                            })
                
                # Search for photos as backup
                photo_url = f"{self.pexels_base_url}/search"
                params = {'query': query, 'per_page': 5}
                
                async with session.get(photo_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        for photo in data.get('photos', [])[:2]:
                            clips.append({
                                'type': 'image',
                                'url': photo['src']['large'],
                                'duration': 5,  # Default duration for images
                                'source': 'pexels',
                                'id': photo['id']
                            })
                            
        except Exception as e:
            logger.error(f"Pexels API error: {e}")
        
        return clips
    
    async def _search_pixabay(self, query: str) -> List[Dict]:
        """Search Pixabay API for videos and images"""
        clips = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Search for videos
                params = {
                    'key': self.pixabay_api_key,
                    'q': query,
                    'video_type': 'all',
                    'per_page': 5
                }
                
                video_url = f"{self.pixabay_base_url}/videos/"
                async with session.get(video_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        for video in data.get('hits', [])[:3]:
                            if 'videos' in video and 'medium' in video['videos']:
                                clips.append({
                                    'type': 'video',
                                    'url': video['videos']['medium'].get('url', ''),
                                    'duration': video['videos']['medium'].get('duration', 10),
                                    'source': 'pixabay',
                                    'id': video['id']
                                })
                
                # Search for images
                params['video_type'] = None  # Remove video_type for image search
                image_url = self.pixabay_base_url
                
                async with session.get(image_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        for image in data.get('hits', [])[:2]:
                            clips.append({
                                'type': 'image',
                                'url': image['largeImageURL'],
                                'duration': 5,
                                'source': 'pixabay',
                                'id': image['id']
                            })
                            
        except Exception as e:
            logger.error(f"Pixabay API error: {e}")
        
        return clips
    
    def _generate_mock_footage(self, keywords: List[str], duration: float) -> List[Dict]:
        """Generate mock footage data for testing"""
        clips = []
        num_clips = max(1, int(duration / 5))  # One clip per 5 seconds
        
        for i in range(num_clips):
            clips.append({
                'type': 'image' if i % 2 == 0 else 'video',
                'path': f"mock_{keywords[0] if keywords else 'stock'}_{i}.jpg",
                'duration': duration / num_clips,
                'source': 'mock',
                'id': f"mock_{i}",
                'keywords': keywords
            })
        
        return clips
    
    def apply_motion_effects(self, clips: List[Dict]) -> List[Dict]:
        """
        Add Ken Burns movement effects to static images/videos
        """
        processed_clips = []
        
        for clip in clips:
            effect = random.choice(self.motion_effects)
            
            processed_clip = clip.copy()
            processed_clip['effect'] = effect
            processed_clip['effect_params'] = self._get_effect_params(effect)
            
            processed_clips.append(processed_clip)
        
        return processed_clips
    
    def _get_effect_params(self, effect: str) -> Dict:
        """Get parameters for motion effect"""
        params = {
            'zoom_in_slow': {
                'start_scale': 1.0,
                'end_scale': 1.2,
                'anchor': 'center'
            },
            'zoom_out_slow': {
                'start_scale': 1.2,
                'end_scale': 1.0,
                'anchor': 'center'
            },
            'pan_left_right': {
                'start_x': -0.1,
                'end_x': 0.1,
                'y': 0
            },
            'pan_top_bottom': {
                'x': 0,
                'start_y': -0.1,
                'end_y': 0.1
            },
            'rotate_subtle': {
                'start_rotation': -2,
                'end_rotation': 2,
                'anchor': 'center'
            }
        }
        
        return params.get(effect, {})
    
    def _is_cache_valid(self, cached_data: Dict) -> bool:
        """Check if cached data is still valid (24 hours)"""
        if 'cached_at' not in cached_data:
            return False
        
        cached_time = datetime.fromisoformat(cached_data['cached_at'])
        age = datetime.utcnow() - cached_time
        
        return age < timedelta(hours=24)
    
    async def download_footage(self, url: str, output_path: str) -> bool:
        """Download footage from URL to local path"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        async with aiofiles.open(output_path, 'wb') as f:
                            await f.write(content)
                        
                        return True
        except Exception as e:
            logger.error(f"Failed to download footage from {url}: {e}")
        
        return False
    
    def get_footage_ratio(self, content_type: str) -> Dict[str, Any]:
        """
        Get optimal footage ratio based on content type
        According to architecture document specifications
        """
        ratios = {
            'educational': {
                'stock': 0.2,
                'generated': 0.8,
                'method': 'slide_based_motion'
            },
            'news_commentary': {
                'stock': 0.9,
                'generated': 0.1,
                'method': 'stock_footage_assembly'
            },
            'tutorial': {
                'stock': 0.3,
                'generated': 0.7,
                'method': 'screen_recording_hybrid'
            },
            'entertainment': {
                'stock': 0.4,
                'generated': 0.6,
                'method': 'ai_generated_compilation'
            }
        }
        
        return ratios.get(content_type, ratios['news_commentary'])


# Global instance
stock_footage_service = StockFootageService()