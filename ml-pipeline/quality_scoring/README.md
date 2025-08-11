# YTEmpire AI/ML Content Quality Scoring System

Advanced multi-modal quality assessment system for video content using computer vision, natural language processing, and audio analysis.

## Overview

The Content Quality Scoring system provides comprehensive analysis of video quality across multiple dimensions:

- **Visual Quality**: Resolution, sharpness, brightness, contrast, color balance, stability
- **Audio Quality**: Clarity, volume consistency, noise levels, speech quality  
- **Content Quality**: Script coherence, topic relevance, engagement potential, information density
- **Technical Quality**: Encoding quality, file size efficiency, duration appropriateness

## Architecture

### Core Components

- **ContentQualityScorer**: Main orchestrator for quality analysis
- **VisualQualityAnalyzer**: Computer vision-based visual analysis using CLIP
- **AudioQualityAnalyzer**: Audio processing using Whisper and librosa
- **ContentQualityAnalyzer**: NLP analysis using BERT, RoBERTa, and BART
- **TechnicalQualityAnalyzer**: Technical video properties analysis
- **QualityScoreDatabase**: SQLite/PostgreSQL storage with caching

### ML Models Integration

- **OpenAI Whisper**: Speech recognition and audio quality assessment
- **OpenAI CLIP**: Visual-semantic understanding and quality evaluation
- **BERT**: Content coherence and semantic analysis
- **RoBERTa**: Sentiment analysis for engagement prediction
- **BART**: Zero-shot topic classification for relevance scoring

## Installation

### Prerequisites

- Python 3.9+ 
- FFmpeg for video/audio processing
- Docker and Docker Compose
- NVIDIA GPU (optional, for acceleration)

### Quick Setup

1. **Run the automated setup script:**
   ```bash
   ./scripts/ml/setup-quality-scoring.sh
   ```

2. **Or manual installation:**
   ```bash
   cd ml-pipeline/quality_scoring
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Start services with Docker:**
   ```bash
   docker-compose -f docker-compose.quality.yml up -d
   ```

## Usage

### API Service

Start the FastAPI service:
```bash
cd ml-pipeline/quality_scoring
source venv/bin/activate
python quality_api.py
```

API will be available at `http://localhost:8001`

### Programmatic Usage

```python
import asyncio
from quality_scorer import ContentQualityScorer, QualityScoringConfig

# Initialize scorer
config = QualityScoringConfig(
    min_acceptable_score=0.65,
    excellent_score_threshold=0.85
)
scorer = ContentQualityScorer(config)

# Analyze video
async def analyze_video():
    metrics = await scorer.score_video(
        video_path="path/to/video.mp4",
        script_text="Your video script here",
        target_topic="technology"
    )
    
    print(f"Overall Quality Score: {metrics.overall_quality_score:.3f}")
    
    # Generate detailed report
    report = scorer.get_quality_report(metrics)
    print(report['overall_assessment']['recommendation'])

# Run analysis
asyncio.run(analyze_video())
```

### API Endpoints

#### Analyze Single Video
```bash
curl -X POST "http://localhost:8001/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/video.mp4",
    "script_text": "Your video script",
    "target_topic": "technology",
    "use_cache": true
  }'
```

#### Batch Analysis
```bash
curl -X POST "http://localhost:8001/analyze/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "video_paths": ["/path/to/video1.mp4", "/path/to/video2.mp4"],
    "script_texts": ["Script 1", "Script 2"],
    "target_topics": ["tech", "education"]
  }'
```

#### Upload and Analyze
```bash
curl -X POST "http://localhost:8001/upload-and-analyze" \
  -F "file=@/path/to/video.mp4" \
  -F "script_text=Your script here" \
  -F "target_topic=technology"
```

#### Get Statistics
```bash
curl "http://localhost:8001/stats"
```

## Quality Metrics

### Visual Quality (25% weight)
- **Resolution Score**: Based on video dimensions (4K=1.0, 1080p=0.9, 720p=0.7)
- **Sharpness Score**: Using Laplacian variance for edge detection
- **Brightness Score**: Optimal brightness around 50% luminance
- **Contrast Score**: Standard deviation of pixel intensities
- **Color Balance Score**: RGB channel balance analysis
- **Visual Stability Score**: Frame-to-frame consistency

### Audio Quality (25% weight)
- **Audio Clarity**: Spectral centroid analysis for sound quality
- **Volume Consistency**: RMS energy variance across audio
- **Noise Level**: Zero-crossing rate for noise detection
- **Speech Quality**: Whisper confidence and word density analysis

### Content Quality (30% weight)
- **Script Coherence**: Semantic similarity between sentences using BERT
- **Topic Relevance**: Zero-shot classification against target topic
- **Engagement Potential**: Sentiment, question count, lexical diversity
- **Information Density**: Ratio of informative words (nouns, verbs, adjectives)

### Technical Quality (20% weight)
- **Encoding Quality**: Codec efficiency and bitrate optimization
- **File Size Efficiency**: Size-to-quality ratio analysis
- **Duration Appropriateness**: Optimal length for content type

## Configuration

Configuration is managed through `config.yaml`:

```yaml
# Model Configuration
models:
  whisper:
    model_size: "base"  # tiny, base, small, medium, large
  clip:
    model_name: "openai/clip-vit-base-patch32"
  bert:
    model_name: "bert-base-uncased"

# Quality Thresholds
quality_thresholds:
  min_acceptable_score: 0.65
  excellent_score_threshold: 0.85

# Processing Configuration
processing:
  max_concurrent_analyses: 4
  frame_sample_rate: 30
  target_processing_time: 120  # seconds

# Feature Weights
scoring_weights:
  visual_weight: 0.25
  audio_weight: 0.25
  content_weight: 0.30
  technical_weight: 0.20
```

## Async Task Processing

Quality scoring supports background processing using Celery:

```bash
# Start Celery worker
celery -A quality_tasks worker --loglevel=info --concurrency=2

# Start Celery beat (for scheduled tasks)
celery -A quality_tasks beat --loglevel=info
```

### Task Examples

```python
from quality_tasks import analyze_video_quality_task

# Submit async task
task = analyze_video_quality_task.delay(
    video_path="/path/to/video.mp4",
    script_text="Your script",
    target_topic="technology"
)

# Get result
result = task.get(timeout=300)
print(f"Quality Score: {result['overall_score']}")
```

## Monitoring and Observability

### Metrics Collection
- Processing times per analysis type
- Quality score distributions  
- Model inference latencies
- Error rates and failure types
- Cache hit rates

### Dashboards
- **Grafana Dashboard**: http://localhost:3001
- **Prometheus Metrics**: http://localhost:9091
- Real-time quality trends
- Performance monitoring
- System resource usage

### Health Checks
```bash
# API health
curl http://localhost:8001/health

# Model status
curl http://localhost:8001/models/status

# Database statistics
curl http://localhost:8001/stats
```

## Performance Optimization

### GPU Acceleration
- Automatic GPU detection and usage
- CUDA optimization for PyTorch models
- Memory management for large models

### Caching Strategy
- Video hash-based result caching
- Configurable cache duration
- Database-backed persistence

### Batch Processing
- Concurrent analysis support
- Resource pooling for efficiency
- Automatic retry on failures

## Quality Scoring Examples

### High Quality Video (Score: 0.92)
```json
{
  "overall_quality_score": 0.92,
  "visual_quality": {
    "resolution": 1.0,      // 4K resolution
    "sharpness": 0.95,      // Very sharp
    "brightness": 0.88,     // Well balanced
    "contrast": 0.91        // Good contrast
  },
  "audio_quality": {
    "clarity": 0.94,        // Clear speech
    "speech_quality": 0.89, // High Whisper confidence
    "noise_level": 0.96     // Low noise
  },
  "content_quality": {
    "coherence": 0.87,      // Logical flow
    "engagement": 0.91,     // High engagement potential
    "relevance": 0.94       // On-topic content
  }
}
```

### Low Quality Video (Score: 0.43)
```json
{
  "overall_quality_score": 0.43,
  "visual_quality": {
    "resolution": 0.3,      // 480p resolution
    "sharpness": 0.42,      // Blurry
    "brightness": 0.31,     // Too dark
    "stability": 0.28       // Shaky footage
  },
  "audio_quality": {
    "clarity": 0.35,        // Muffled audio
    "noise_level": 0.22,    // High background noise
    "volume": 0.45          // Inconsistent volume
  },
  "recommendations": [
    "Improve lighting conditions",
    "Use image stabilization",
    "Reduce background noise",
    "Increase video resolution"
  ]
}
```

## Integration with YTEmpire Pipeline

Quality scoring integrates seamlessly with the video generation pipeline:

1. **Pre-upload Analysis**: Score videos before YouTube upload
2. **Quality Gates**: Reject videos below minimum score threshold
3. **Optimization Feedback**: Provide specific improvement recommendations
4. **A/B Testing**: Compare quality scores across different generation approaches
5. **Performance Tracking**: Monitor quality trends over time

## Development and Testing

### Running Tests
```bash
cd ml-pipeline/quality_scoring
source venv/bin/activate

# Unit tests
python -m pytest test/ -v

# Integration tests
python test_quality_scorer.py

# Generate test video
python ../scripts/ml/generate-test-video.py
```

### Adding Custom Metrics

1. Extend the `VideoAnalysisMetrics` dataclass
2. Implement analysis logic in appropriate analyzer
3. Update composite scoring weights
4. Add to configuration file

### Model Fine-tuning

The system supports custom model integration:
- Replace model paths in configuration
- Implement custom analyzer classes
- Maintain interface compatibility

## Troubleshooting

### Common Issues

**Out of Memory Errors**
- Reduce `max_concurrent_analyses` in config
- Use smaller model variants (Whisper "tiny" vs "base")
- Enable model quantization

**Slow Processing**
- Enable GPU acceleration
- Reduce frame sampling rate
- Use model caching
- Optimize Docker resource allocation

**Model Download Failures**
- Check internet connection
- Verify Hugging Face Hub access
- Use local model paths if needed

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run analysis with verbose output
config.debug_mode = True
metrics = await scorer.score_video(video_path, verbose=True)
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/quality-improvement`)
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## License

This component is part of the YTEmpire project and follows the same licensing terms.

---

For more information, see the main YTEmpire documentation or contact the AI/ML team.