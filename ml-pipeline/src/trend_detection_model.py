"""
Trend Detection Model for YouTube Content
Uses multiple data sources to identify trending topics and predict viral potential
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
import joblib
import json
import logging
from dataclasses import dataclass
from enum import Enum
import redis
from collections import defaultdict
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendSource(Enum):
    """Trend data sources"""
    YOUTUBE = "youtube"
    GOOGLE_TRENDS = "google_trends"
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"

@dataclass
class TrendingTopic:
    """Trending topic data structure"""
    topic: str
    score: float  # 0-100 trend score
    category: str
    keywords: List[str]
    competition_level: str  # low, medium, high
    predicted_views: int
    best_time_to_post: datetime
    source_scores: Dict[str, float]
    metadata: Dict[str, Any]

class TrendDetectionModel:
    """
    Advanced trend detection model combining multiple ML techniques
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.rf_model = None
        self.gb_model = None
        self.neural_model = None
        self.scaler = StandardScaler()
        self.tokenizer = None
        self.bert_model = None
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.cache_ttl = 3600  # 1 hour cache
        
        # Initialize models
        self._initialize_models()
        
        # Load pre-trained models if available
        if model_path:
            self.load_models(model_path)
    
    def _initialize_models(self):
        """Initialize ML models"""
        # Random Forest for trend scoring
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        # Gradient Boosting for view prediction
        self.gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # Neural network for complex pattern recognition
        self.neural_model = TrendNeuralNetwork(
            input_size=50,
            hidden_size=128,
            output_size=1
        )
        
        # BERT for text analysis
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
    
    async def detect_trends(
        self,
        category: str,
        region: str = "US",
        time_range: str = "now",
        limit: int = 10
    ) -> List[TrendingTopic]:
        """
        Detect trending topics for a specific category
        
        Args:
            category: Content category (gaming, tech, etc.)
            region: Geographic region
            time_range: Time range for trend analysis
            limit: Number of trends to return
        
        Returns:
            List of trending topics with scores and metadata
        """
        # Check cache first
        cache_key = f"trends:{category}:{region}:{time_range}"
        cached_data = self.redis_client.get(cache_key)
        
        if cached_data:
            return self._deserialize_trends(cached_data)
        
        # Collect data from multiple sources
        trend_data = await self._collect_trend_data(category, region, time_range)
        
        # Extract features
        features = self._extract_features(trend_data)
        
        # Score trends using ensemble model
        trend_scores = self._score_trends(features)
        
        # Predict performance metrics
        predictions = self._predict_performance(features)
        
        # Compile trending topics
        trending_topics = self._compile_trending_topics(
            trend_data,
            trend_scores,
            predictions,
            category
        )
        
        # Sort by score and limit
        trending_topics.sort(key=lambda x: x.score, reverse=True)
        trending_topics = trending_topics[:limit]
        
        # Cache results
        self.redis_client.setex(
            cache_key,
            self.cache_ttl,
            self._serialize_trends(trending_topics)
        )
        
        return trending_topics
    
    async def _collect_trend_data(
        self,
        category: str,
        region: str,
        time_range: str
    ) -> Dict[str, Any]:
        """Collect trend data from multiple sources"""
        tasks = [
            self._fetch_youtube_trends(category, region),
            self._fetch_google_trends(category, region, time_range),
            self._fetch_social_trends(category),
            self._fetch_news_trends(category)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        trend_data = {
            'youtube': results[0] if not isinstance(results[0], Exception) else [],
            'google': results[1] if not isinstance(results[1], Exception) else [],
            'social': results[2] if not isinstance(results[2], Exception) else [],
            'news': results[3] if not isinstance(results[3], Exception) else []
        }
        
        return trend_data
    
    async def _fetch_youtube_trends(self, category: str, region: str) -> List[Dict]:
        """Fetch trending videos from YouTube"""
        # Simulated YouTube API call
        # In production, use actual YouTube Data API
        trends = []
        
        try:
            # Mock data for demonstration
            sample_trends = [
                {
                    'title': f'{category} trend 1',
                    'view_count': 1000000,
                    'like_count': 50000,
                    'comment_count': 5000,
                    'published_at': datetime.now() - timedelta(days=1)
                },
                {
                    'title': f'{category} trend 2',
                    'view_count': 500000,
                    'like_count': 25000,
                    'comment_count': 2500,
                    'published_at': datetime.now() - timedelta(days=2)
                }
            ]
            
            for trend in sample_trends:
                # Calculate engagement rate
                engagement_rate = (trend['like_count'] + trend['comment_count']) / trend['view_count']
                trend['engagement_rate'] = engagement_rate
                
                # Calculate velocity (views per hour since published)
                hours_since_published = (datetime.now() - trend['published_at']).total_seconds() / 3600
                trend['velocity'] = trend['view_count'] / max(hours_since_published, 1)
                
                trends.append(trend)
                
        except Exception as e:
            logger.error(f"Error fetching YouTube trends: {e}")
        
        return trends
    
    async def _fetch_google_trends(
        self,
        category: str,
        region: str,
        time_range: str
    ) -> List[Dict]:
        """Fetch Google Trends data"""
        # Simulated Google Trends API call
        # In production, use pytrends library
        trends = []
        
        try:
            # Mock data
            sample_keywords = [
                {'keyword': f'{category} tutorial', 'interest': 85},
                {'keyword': f'best {category} 2024', 'interest': 72},
                {'keyword': f'{category} tips', 'interest': 68}
            ]
            
            for kw in sample_keywords:
                trends.append({
                    'keyword': kw['keyword'],
                    'interest_score': kw['interest'],
                    'rising': kw['interest'] > 70
                })
                
        except Exception as e:
            logger.error(f"Error fetching Google Trends: {e}")
        
        return trends
    
    async def _fetch_social_trends(self, category: str) -> List[Dict]:
        """Fetch trends from social media"""
        # Simulated social media API calls
        trends = []
        
        try:
            # Mock Twitter/Reddit data
            sample_topics = [
                {
                    'topic': f'#{category}tips',
                    'mentions': 10000,
                    'sentiment': 0.75,
                    'platform': 'twitter'
                },
                {
                    'topic': f'r/{category}',
                    'posts': 500,
                    'upvotes': 15000,
                    'platform': 'reddit'
                }
            ]
            
            trends.extend(sample_topics)
            
        except Exception as e:
            logger.error(f"Error fetching social trends: {e}")
        
        return trends
    
    async def _fetch_news_trends(self, category: str) -> List[Dict]:
        """Fetch trending news topics"""
        # Simulated news API call
        trends = []
        
        try:
            # Mock news data
            sample_news = [
                {
                    'headline': f'Breaking: New {category} innovation',
                    'source': 'TechNews',
                    'published': datetime.now() - timedelta(hours=3),
                    'relevance': 0.9
                }
            ]
            
            trends.extend(sample_news)
            
        except Exception as e:
            logger.error(f"Error fetching news trends: {e}")
        
        return trends
    
    def _extract_features(self, trend_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from trend data"""
        features = []
        
        # YouTube features
        youtube_features = self._extract_youtube_features(trend_data.get('youtube', []))
        features.extend(youtube_features)
        
        # Google Trends features
        google_features = self._extract_google_features(trend_data.get('google', []))
        features.extend(google_features)
        
        # Social media features
        social_features = self._extract_social_features(trend_data.get('social', []))
        features.extend(social_features)
        
        # News features
        news_features = self._extract_news_features(trend_data.get('news', []))
        features.extend(news_features)
        
        # Pad or truncate to fixed size
        feature_size = 50
        if len(features) < feature_size:
            features.extend([0] * (feature_size - len(features)))
        else:
            features = features[:feature_size]
        
        return np.array(features).reshape(1, -1)
    
    def _extract_youtube_features(self, youtube_data: List[Dict]) -> List[float]:
        """Extract features from YouTube data"""
        if not youtube_data:
            return [0] * 10
        
        features = []
        
        # Average metrics
        avg_views = np.mean([d.get('view_count', 0) for d in youtube_data])
        avg_engagement = np.mean([d.get('engagement_rate', 0) for d in youtube_data])
        avg_velocity = np.mean([d.get('velocity', 0) for d in youtube_data])
        
        # Normalize to 0-1 scale
        features.append(min(avg_views / 1000000, 1))  # Normalize by 1M views
        features.append(min(avg_engagement * 100, 1))  # Engagement rate
        features.append(min(avg_velocity / 10000, 1))  # Velocity
        
        # Trend indicators
        features.append(1 if any(d.get('velocity', 0) > 5000 for d in youtube_data) else 0)
        features.append(len(youtube_data) / 10)  # Number of trending videos
        
        # Fill remaining
        features.extend([0] * (10 - len(features)))
        
        return features[:10]
    
    def _extract_google_features(self, google_data: List[Dict]) -> List[float]:
        """Extract features from Google Trends data"""
        if not google_data:
            return [0] * 10
        
        features = []
        
        # Interest scores
        max_interest = max([d.get('interest_score', 0) for d in google_data], default=0)
        avg_interest = np.mean([d.get('interest_score', 0) for d in google_data])
        rising_count = sum(1 for d in google_data if d.get('rising', False))
        
        features.append(max_interest / 100)
        features.append(avg_interest / 100)
        features.append(rising_count / max(len(google_data), 1))
        
        # Fill remaining
        features.extend([0] * (10 - len(features)))
        
        return features[:10]
    
    def _extract_social_features(self, social_data: List[Dict]) -> List[float]:
        """Extract features from social media data"""
        if not social_data:
            return [0] * 10
        
        features = []
        
        # Social metrics
        total_mentions = sum(d.get('mentions', 0) for d in social_data)
        avg_sentiment = np.mean([d.get('sentiment', 0) for d in social_data])
        
        features.append(min(total_mentions / 50000, 1))
        features.append((avg_sentiment + 1) / 2)  # Normalize sentiment to 0-1
        
        # Platform diversity
        platforms = set(d.get('platform') for d in social_data)
        features.append(len(platforms) / 5)  # Assume 5 major platforms
        
        # Fill remaining
        features.extend([0] * (10 - len(features)))
        
        return features[:10]
    
    def _extract_news_features(self, news_data: List[Dict]) -> List[float]:
        """Extract features from news data"""
        if not news_data:
            return [0] * 10
        
        features = []
        
        # News metrics
        news_count = len(news_data)
        avg_relevance = np.mean([d.get('relevance', 0) for d in news_data])
        
        # Recency score
        recent_news = sum(
            1 for d in news_data 
            if (datetime.now() - d.get('published', datetime.min)).total_seconds() < 86400
        )
        
        features.append(min(news_count / 10, 1))
        features.append(avg_relevance)
        features.append(recent_news / max(news_count, 1))
        
        # Fill remaining
        features.extend([0] * (10 - len(features)))
        
        return features[:10]
    
    def _score_trends(self, features: np.ndarray) -> np.ndarray:
        """Score trends using ensemble model"""
        scores = []
        
        # Random Forest prediction
        if self.rf_model:
            try:
                rf_score = self.rf_model.predict(features)[0]
                scores.append(rf_score)
            except:
                scores.append(50)  # Default score
        
        # Neural network prediction
        if self.neural_model:
            try:
                nn_score = self._neural_predict(features)
                scores.append(nn_score)
            except:
                scores.append(50)
        
        # Combine scores (weighted average)
        if scores:
            final_score = np.mean(scores)
        else:
            # Fallback to rule-based scoring
            final_score = self._rule_based_scoring(features)
        
        return np.array([min(max(final_score, 0), 100)])  # Clamp to 0-100
    
    def _neural_predict(self, features: np.ndarray) -> float:
        """Make prediction using neural network"""
        self.neural_model.eval()
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(features)
            output = self.neural_model(input_tensor)
            score = output.item() * 100  # Scale to 0-100
        
        return score
    
    def _rule_based_scoring(self, features: np.ndarray) -> float:
        """Fallback rule-based scoring"""
        # Simple weighted sum of features
        weights = np.array([
            0.3,  # YouTube views
            0.2,  # Engagement
            0.15,  # Velocity
            0.15,  # Google interest
            0.1,  # Social mentions
            0.1   # News relevance
        ])
        
        # Take first 6 features for simplicity
        score = np.dot(features[0][:6], weights) * 100
        
        return score
    
    def _predict_performance(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict video performance metrics"""
        predictions = {
            'predicted_views': 10000,  # Default
            'predicted_engagement': 0.05,
            'predicted_ctr': 0.10,
            'optimal_duration': 600,  # 10 minutes
            'best_time_to_post': self._calculate_best_posting_time()
        }
        
        if self.gb_model:
            try:
                # Predict views (log scale)
                log_views = self.gb_model.predict(features)[0]
                predictions['predicted_views'] = int(np.exp(log_views))
            except:
                pass
        
        return predictions
    
    def _calculate_best_posting_time(self) -> datetime:
        """Calculate optimal posting time"""
        # Best times: Tuesday-Thursday, 2-4 PM EST
        now = datetime.now()
        
        # Find next Tuesday-Thursday
        days_ahead = 0
        while now.weekday() not in [1, 2, 3]:  # Tuesday, Wednesday, Thursday
            days_ahead += 1
            now += timedelta(days=1)
        
        # Set time to 3 PM
        best_time = now.replace(hour=15, minute=0, second=0, microsecond=0)
        
        return best_time
    
    def _compile_trending_topics(
        self,
        trend_data: Dict[str, Any],
        trend_scores: np.ndarray,
        predictions: Dict[str, Any],
        category: str
    ) -> List[TrendingTopic]:
        """Compile trending topics from analysis results"""
        trending_topics = []
        
        # Extract unique topics from all sources
        all_topics = self._extract_unique_topics(trend_data)
        
        for i, topic in enumerate(all_topics[:10]):  # Limit to 10 topics
            # Calculate competition level
            competition = self._assess_competition(topic, trend_data)
            
            # Extract keywords
            keywords = self._extract_keywords(topic, trend_data)
            
            # Create trending topic object
            trending_topic = TrendingTopic(
                topic=topic,
                score=float(trend_scores[0]) if i == 0 else float(trend_scores[0] * (0.9 ** i)),
                category=category,
                keywords=keywords,
                competition_level=competition,
                predicted_views=predictions['predicted_views'] // (i + 1),
                best_time_to_post=predictions['best_time_to_post'],
                source_scores={
                    'youtube': self._calculate_source_score(topic, trend_data['youtube']),
                    'google': self._calculate_source_score(topic, trend_data['google']),
                    'social': self._calculate_source_score(topic, trend_data['social']),
                    'news': self._calculate_source_score(topic, trend_data['news'])
                },
                metadata={
                    'predicted_engagement': predictions['predicted_engagement'],
                    'predicted_ctr': predictions['predicted_ctr'],
                    'optimal_duration': predictions['optimal_duration']
                }
            )
            
            trending_topics.append(trending_topic)
        
        return trending_topics
    
    def _extract_unique_topics(self, trend_data: Dict[str, Any]) -> List[str]:
        """Extract unique topics from all data sources"""
        topics = set()
        
        # From YouTube
        for item in trend_data.get('youtube', []):
            if 'title' in item:
                # Extract main topic from title
                topics.add(self._extract_topic_from_text(item['title']))
        
        # From Google
        for item in trend_data.get('google', []):
            if 'keyword' in item:
                topics.add(item['keyword'])
        
        # From Social
        for item in trend_data.get('social', []):
            if 'topic' in item:
                topics.add(item['topic'].replace('#', '').replace('r/', ''))
        
        # From News
        for item in trend_data.get('news', []):
            if 'headline' in item:
                topics.add(self._extract_topic_from_text(item['headline']))
        
        return list(topics)
    
    def _extract_topic_from_text(self, text: str) -> str:
        """Extract main topic from text"""
        # Simple extraction - in production, use NLP
        # Remove common words and extract key phrase
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 'or', 'but'}
        words = text.lower().split()
        key_words = [w for w in words if w not in stop_words]
        
        # Return first 3 key words as topic
        return ' '.join(key_words[:3])
    
    def _assess_competition(self, topic: str, trend_data: Dict[str, Any]) -> str:
        """Assess competition level for a topic"""
        # Count occurrences across sources
        occurrences = 0
        
        for source_data in trend_data.values():
            for item in source_data:
                if topic.lower() in str(item).lower():
                    occurrences += 1
        
        # Determine competition level
        if occurrences > 10:
            return 'high'
        elif occurrences > 5:
            return 'medium'
        else:
            return 'low'
    
    def _extract_keywords(self, topic: str, trend_data: Dict[str, Any]) -> List[str]:
        """Extract relevant keywords for a topic"""
        keywords = set()
        
        # Add topic words
        keywords.update(topic.lower().split())
        
        # Find related terms from data
        for source_data in trend_data.values():
            for item in source_data:
                if topic.lower() in str(item).lower():
                    # Extract additional keywords from matching items
                    if isinstance(item, dict):
                        for value in item.values():
                            if isinstance(value, str):
                                words = value.lower().split()
                                keywords.update(w for w in words if len(w) > 3)
        
        # Limit to top 10 keywords
        return list(keywords)[:10]
    
    def _calculate_source_score(self, topic: str, source_data: List[Dict]) -> float:
        """Calculate topic score for a specific source"""
        if not source_data:
            return 0.0
        
        score = 0.0
        matches = 0
        
        for item in source_data:
            if topic.lower() in str(item).lower():
                matches += 1
                # Add source-specific scoring
                if 'interest_score' in item:
                    score += item['interest_score'] / 100
                elif 'engagement_rate' in item:
                    score += item['engagement_rate'] * 100
                elif 'relevance' in item:
                    score += item['relevance'] * 100
                else:
                    score += 50  # Default score for match
        
        return min(score / max(matches, 1), 100)
    
    def _serialize_trends(self, trends: List[TrendingTopic]) -> str:
        """Serialize trending topics for caching"""
        serialized = []
        
        for trend in trends:
            serialized.append({
                'topic': trend.topic,
                'score': trend.score,
                'category': trend.category,
                'keywords': trend.keywords,
                'competition_level': trend.competition_level,
                'predicted_views': trend.predicted_views,
                'best_time_to_post': trend.best_time_to_post.isoformat(),
                'source_scores': trend.source_scores,
                'metadata': trend.metadata
            })
        
        return json.dumps(serialized)
    
    def _deserialize_trends(self, data: str) -> List[TrendingTopic]:
        """Deserialize trending topics from cache"""
        deserialized = json.loads(data)
        trends = []
        
        for item in deserialized:
            trend = TrendingTopic(
                topic=item['topic'],
                score=item['score'],
                category=item['category'],
                keywords=item['keywords'],
                competition_level=item['competition_level'],
                predicted_views=item['predicted_views'],
                best_time_to_post=datetime.fromisoformat(item['best_time_to_post']),
                source_scores=item['source_scores'],
                metadata=item['metadata']
            )
            trends.append(trend)
        
        return trends
    
    def train(self, training_data: pd.DataFrame):
        """Train the trend detection models"""
        # Prepare features and targets
        X = self._prepare_training_features(training_data)
        y_score = training_data['trend_score'].values
        y_views = np.log(training_data['actual_views'].values + 1)
        
        # Split data
        X_train, X_test, y_score_train, y_score_test = train_test_split(
            X, y_score, test_size=0.2, random_state=42
        )
        
        _, _, y_views_train, y_views_test = train_test_split(
            X, y_views, test_size=0.2, random_state=42
        )
        
        # Train Random Forest for scoring
        self.rf_model.fit(X_train, y_score_train)
        rf_score = self.rf_model.score(X_test, y_score_test)
        logger.info(f"Random Forest R² score: {rf_score:.3f}")
        
        # Train Gradient Boosting for view prediction
        self.gb_model.fit(X_train, y_views_train)
        gb_score = self.gb_model.score(X_test, y_views_test)
        logger.info(f"Gradient Boosting R² score: {gb_score:.3f}")
        
        # Train Neural Network
        self._train_neural_network(X_train, y_score_train, X_test, y_score_test)
    
    def _prepare_training_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features from training data"""
        # Extract features from dataframe
        # This should match the feature extraction in detect_trends
        features = []
        
        for _, row in data.iterrows():
            row_features = []
            
            # Add your feature extraction logic here
            # Example features:
            row_features.append(row.get('youtube_views', 0) / 1000000)
            row_features.append(row.get('engagement_rate', 0))
            row_features.append(row.get('google_interest', 0) / 100)
            row_features.append(row.get('social_mentions', 0) / 10000)
            
            # Pad to 50 features
            row_features.extend([0] * (50 - len(row_features)))
            features.append(row_features[:50])
        
        return np.array(features)
    
    def _train_neural_network(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 100
    ):
        """Train the neural network model"""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1))
        
        # Define optimizer and loss
        optimizer = optim.Adam(self.neural_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop
        self.neural_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.neural_model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor / 100)  # Normalize to 0-1
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                self.neural_model.eval()
                with torch.no_grad():
                    test_outputs = self.neural_model(X_test_tensor)
                    test_loss = criterion(test_outputs, y_test_tensor / 100)
                    logger.info(f"Epoch {epoch}, Test Loss: {test_loss:.4f}")
                self.neural_model.train()
    
    def save_models(self, path: str):
        """Save trained models to disk"""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save sklearn models
        joblib.dump(self.rf_model, f"{path}/rf_model.pkl")
        joblib.dump(self.gb_model, f"{path}/gb_model.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        
        # Save neural network
        torch.save(self.neural_model.state_dict(), f"{path}/neural_model.pt")
        
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """Load trained models from disk"""
        try:
            self.rf_model = joblib.load(f"{path}/rf_model.pkl")
            self.gb_model = joblib.load(f"{path}/gb_model.pkl")
            self.scaler = joblib.load(f"{path}/scaler.pkl")
            
            self.neural_model.load_state_dict(torch.load(f"{path}/neural_model.pt"))
            self.neural_model.eval()
            
            logger.info(f"Models loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")


class TrendNeuralNetwork(nn.Module):
    """Neural network for trend prediction"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(TrendNeuralNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x


# API endpoint wrapper
class TrendDetectionAPI:
    """API wrapper for trend detection model"""
    
    def __init__(self):
        self.model = TrendDetectionModel(model_path="models/trend_detection")
    
    async def get_trending_topics(
        self,
        category: str,
        region: str = "US",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get trending topics for API response"""
        trends = await self.model.detect_trends(category, region, "now", limit)
        
        # Convert to API response format
        response = []
        for trend in trends:
            response.append({
                'topic': trend.topic,
                'score': trend.score,
                'keywords': trend.keywords,
                'competition_level': trend.competition_level,
                'predicted_views': trend.predicted_views,
                'best_time_to_post': trend.best_time_to_post.isoformat(),
                'metadata': trend.metadata
            })
        
        return response


# Initialize global model instance
trend_detection_api = TrendDetectionAPI()