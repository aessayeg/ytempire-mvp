"""
Prompt Engineering Framework
Advanced prompt optimization, templates, and management system
"""
import json
import yaml
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import re
import hashlib
from pathlib import Path
import logging
import asyncio
from collections import defaultdict
import numpy as np
from jinja2 import Template, Environment, FileSystemLoader
import tiktoken

logger = logging.getLogger(__name__)

class PromptType(Enum):
    """Types of prompts in the system"""
    SCRIPT_GENERATION = "script_generation"
    TITLE_GENERATION = "title_generation"
    DESCRIPTION_GENERATION = "description_generation"
    THUMBNAIL_GENERATION = "thumbnail_generation"
    HOOK_GENERATION = "hook_generation"
    TAGS_GENERATION = "tags_generation"
    FACT_CHECKING = "fact_checking"
    TONE_ADJUSTMENT = "tone_adjustment"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"

class VideoStyle(Enum):
    """Video content styles"""
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    TUTORIAL = "tutorial"
    NEWS = "news"
    DOCUMENTARY = "documentary"
    MOTIVATIONAL = "motivational"
    COMEDY = "comedy"
    STORYTELLING = "storytelling"
    REVIEW = "review"
    VLOG = "vlog"

class TargetAudience(Enum):
    """Target audience segments"""
    KIDS = "kids"
    TEENS = "teens"
    YOUNG_ADULTS = "young_adults"
    ADULTS = "adults"
    SENIORS = "seniors"
    TECH_SAVVY = "tech_savvy"
    BEGINNERS = "beginners"
    PROFESSIONALS = "professionals"
    ENTHUSIASTS = "enthusiasts"

@dataclass
class PromptTemplate:
    """Template for prompt generation"""
    id: str
    name: str
    type: PromptType
    template: str
    variables: List[str]
    constraints: Dict[str, Any]
    examples: List[Dict[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class PromptOptimizationResult:
    """Result of prompt optimization"""
    original_prompt: str
    optimized_prompt: str
    token_count: int
    estimated_cost: float
    optimization_score: float
    suggestions: List[str]
    metrics: Dict[str, Any]

class PromptEngineeringFramework:
    """Advanced prompt engineering and optimization system"""
    
    def __init__(self, templates_dir: str = "prompts/templates"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.templates: Dict[str, PromptTemplate] = {}
        self.prompt_cache: Dict[str, str] = {}
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.jinja_env = Environment(loader=FileSystemLoader(self.templates_dir))
        self._load_templates()
        self._initialize_default_templates()
        
    def _initialize_default_templates(self):
        """Initialize default prompt templates"""
        
        # Script Generation Template
        self.register_template(PromptTemplate(
            id="script_gen_v1",
            name="Video Script Generation",
            type=PromptType.SCRIPT_GENERATION,
            template="""Create a compelling YouTube video script about {topic}.

Style: {style}
Target Audience: {audience}
Duration: {duration} minutes
Tone: {tone}

Requirements:
1. Start with a strong hook (first 15 seconds)
2. Include {num_key_points} key points
3. Use storytelling elements
4. Include call-to-action
5. Optimize for retention

Additional Context:
{context}

Structure:
- Hook (0:00-0:15)
- Introduction (0:15-0:30)
- Main Content ({num_key_points} sections)
- Summary/Recap
- Call to Action
- End screen suggestion

Make it engaging, informative, and optimized for YouTube's algorithm.""",
            variables=["topic", "style", "audience", "duration", "tone", "num_key_points", "context"],
            constraints={
                "max_tokens": 2000,
                "temperature": 0.7,
                "style_options": [s.value for s in VideoStyle],
                "audience_options": [a.value for a in TargetAudience]
            },
            examples=[
                {
                    "topic": "Artificial Intelligence in 2024",
                    "style": "educational",
                    "audience": "tech_savvy",
                    "duration": "10",
                    "tone": "professional yet accessible",
                    "num_key_points": "5",
                    "context": "Focus on practical applications"
                }
            ],
            metadata={"category": "content_generation", "tier": "premium"}
        ))
        
        # Title Generation Template
        self.register_template(PromptTemplate(
            id="title_gen_v1",
            name="Video Title Generation",
            type=PromptType.TITLE_GENERATION,
            template="""Generate 5 compelling YouTube video titles for: {topic}

Video Style: {style}
Target Keywords: {keywords}
Competitor Titles: {competitor_titles}

Requirements:
- Maximum 60 characters
- Include power words
- Create curiosity gap
- SEO optimized
- Avoid clickbait
- Use numbers when relevant
- Consider emotional triggers

Successful title patterns:
- "How to [achieve desirable outcome] in [timeframe]"
- "[Number] [adjective] Ways to [solve problem]"
- "Why [counterintuitive statement] (And How to [solution])"
- "The [superlative] Guide to [topic]"
- "[Celebrity/Authority] Reveals [secret/method]"

Generate titles that will maximize CTR while maintaining authenticity.""",
            variables=["topic", "style", "keywords", "competitor_titles"],
            constraints={
                "max_length": 60,
                "min_options": 5,
                "include_keywords": True
            },
            examples=[],
            metadata={"category": "metadata_generation"}
        ))
        
        # Hook Generation Template
        self.register_template(PromptTemplate(
            id="hook_gen_v1",
            name="Video Hook Generation",
            type=PromptType.HOOK_GENERATION,
            template="""Create a powerful 15-second hook for a video about: {topic}

Current Title: {title}
Target Emotion: {emotion}
Style: {style}

Hook Types to Consider:
1. Question Hook: Pose an intriguing question
2. Statistic Hook: Share a surprising statistic
3. Story Hook: Start with a brief story
4. Problem Hook: Highlight a problem viewers face
5. Benefit Hook: Promise a specific benefit

Requirements:
- Maximum 50 words
- Create immediate curiosity
- Set clear expectations
- Match video tone
- Include pattern interrupt

Output format:
[VISUAL]: What viewers see
[SCRIPT]: Exact words to say
[EMOTION]: Intended emotional response""",
            variables=["topic", "title", "emotion", "style"],
            constraints={
                "max_words": 50,
                "duration_seconds": 15
            },
            examples=[],
            metadata={"category": "engagement_optimization"}
        ))
        
    def register_template(self, template: PromptTemplate) -> None:
        """Register a new prompt template"""
        self.templates[template.id] = template
        # Save to file for persistence
        self._save_template(template)
        logger.info(f"Registered template: {template.id}")
        
    def _save_template(self, template: PromptTemplate) -> None:
        """Save template to file"""
        file_path = self.templates_dir / f"{template.id}.yaml"
        with open(file_path, 'w') as f:
            yaml.dump({
                'id': template.id,
                'name': template.name,
                'type': template.type.value,
                'template': template.template,
                'variables': template.variables,
                'constraints': template.constraints,
                'examples': template.examples,
                'metadata': template.metadata,
                'version': template.version,
                'performance_metrics': template.performance_metrics
            }, f)
            
    def _load_templates(self) -> None:
        """Load templates from files"""
        for file_path in self.templates_dir.glob("*.yaml"):
            try:
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)
                    template = PromptTemplate(
                        id=data['id'],
                        name=data['name'],
                        type=PromptType(data['type']),
                        template=data['template'],
                        variables=data['variables'],
                        constraints=data['constraints'],
                        examples=data.get('examples', []),
                        metadata=data.get('metadata', {}),
                        version=data.get('version', '1.0.0'),
                        performance_metrics=data.get('performance_metrics', {})
                    )
                    self.templates[template.id] = template
            except Exception as e:
                logger.error(f"Error loading template {file_path}: {e}")
                
    async def generate_prompt(
        self,
        template_id: str,
        variables: Dict[str, Any],
        optimize: bool = True
    ) -> str:
        """Generate a prompt from template"""
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
            
        template = self.templates[template_id]
        
        # Validate required variables
        missing_vars = set(template.variables) - set(variables.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
            
        # Check cache
        cache_key = self._get_cache_key(template_id, variables)
        if cache_key in self.prompt_cache:
            logger.info(f"Using cached prompt for {template_id}")
            return self.prompt_cache[cache_key]
            
        # Render template
        prompt = template.template.format(**variables)
        
        # Optimize if requested
        if optimize:
            optimization_result = await self.optimize_prompt(prompt, template.constraints)
            prompt = optimization_result.optimized_prompt
            
        # Cache the result
        self.prompt_cache[cache_key] = prompt
        
        return prompt
        
    async def optimize_prompt(
        self,
        prompt: str,
        constraints: Dict[str, Any] = None
    ) -> PromptOptimizationResult:
        """Optimize a prompt for better performance and cost"""
        original_prompt = prompt
        optimized_prompt = prompt
        suggestions = []
        
        # Token count and cost estimation
        token_count = len(self.tokenizer.encode(prompt))
        estimated_cost = self._estimate_cost(token_count)
        
        # Optimization techniques
        
        # 1. Remove redundancy
        optimized_prompt = self._remove_redundancy(optimized_prompt)
        if optimized_prompt != prompt:
            suggestions.append("Removed redundant phrases")
            
        # 2. Simplify complex sentences
        optimized_prompt = self._simplify_sentences(optimized_prompt)
        if len(optimized_prompt) < len(prompt) * 0.9:
            suggestions.append("Simplified complex sentences")
            
        # 3. Add structure markers
        if "1." not in optimized_prompt and "bullet" not in optimized_prompt.lower():
            optimized_prompt = self._add_structure_markers(optimized_prompt)
            suggestions.append("Added structure markers for clarity")
            
        # 4. Optimize for token limit
        if constraints and "max_tokens" in constraints:
            max_tokens = constraints["max_tokens"]
            if token_count > max_tokens:
                optimized_prompt = self._truncate_to_token_limit(optimized_prompt, max_tokens)
                suggestions.append(f"Truncated to {max_tokens} tokens")
                
        # 5. Add few-shot examples if beneficial
        if constraints and "examples" in constraints and constraints["examples"]:
            optimized_prompt = self._add_few_shot_examples(optimized_prompt, constraints["examples"])
            suggestions.append("Added few-shot examples")
            
        # Calculate optimization score
        new_token_count = len(self.tokenizer.encode(optimized_prompt))
        optimization_score = self._calculate_optimization_score(
            original_prompt, optimized_prompt, token_count, new_token_count
        )
        
        return PromptOptimizationResult(
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt,
            token_count=new_token_count,
            estimated_cost=self._estimate_cost(new_token_count),
            optimization_score=optimization_score,
            suggestions=suggestions,
            metrics={
                "token_reduction": token_count - new_token_count,
                "cost_savings": estimated_cost - self._estimate_cost(new_token_count),
                "readability_score": self._calculate_readability(optimized_prompt)
            }
        )
        
    def _remove_redundancy(self, text: str) -> str:
        """Remove redundant phrases and words"""
        # Remove duplicate words
        words = text.split()
        cleaned_words = []
        prev_word = ""
        for word in words:
            if word.lower() != prev_word.lower() or word.lower() in ["the", "a", "an", "and", "or"]:
                cleaned_words.append(word)
            prev_word = word
        
        # Remove redundant phrases
        redundant_phrases = [
            "in order to", "at this point in time", "due to the fact that",
            "in the event that", "for the purpose of", "with regard to"
        ]
        text = " ".join(cleaned_words)
        for phrase in redundant_phrases:
            text = text.replace(phrase, self._get_simpler_alternative(phrase))
            
        return text
        
    def _get_simpler_alternative(self, phrase: str) -> str:
        """Get simpler alternative for redundant phrases"""
        alternatives = {
            "in order to": "to",
            "at this point in time": "now",
            "due to the fact that": "because",
            "in the event that": "if",
            "for the purpose of": "for",
            "with regard to": "about"
        }
        return alternatives.get(phrase, phrase)
        
    def _simplify_sentences(self, text: str) -> str:
        """Simplify complex sentences"""
        sentences = text.split('. ')
        simplified = []
        
        for sentence in sentences:
            # Break long sentences
            if len(sentence) > 150:
                # Find natural break points
                if ', and' in sentence:
                    parts = sentence.split(', and')
                    simplified.extend([part.strip() for part in parts])
                elif '; ' in sentence:
                    parts = sentence.split('; ')
                    simplified.extend([part.strip() for part in parts])
                else:
                    simplified.append(sentence)
            else:
                simplified.append(sentence)
                
        return '. '.join(simplified)
        
    def _add_structure_markers(self, text: str) -> str:
        """Add structure markers for clarity"""
        lines = text.split('\n')
        structured = []
        
        for i, line in enumerate(lines):
            if line.strip() and not line.strip()[0].isdigit() and len(line) > 50:
                # Add numbering for long paragraphs
                if i > 0 and lines[i-1].strip() == "":
                    structured.append(f"{i+1}. {line}")
                else:
                    structured.append(line)
            else:
                structured.append(line)
                
        return '\n'.join(structured)
        
    def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to token limit"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = self.tokenizer.decode(tokens)
        return text
        
    def _add_few_shot_examples(self, prompt: str, examples: List[Dict[str, str]]) -> str:
        """Add few-shot examples to prompt"""
        if not examples:
            return prompt
            
        example_text = "\n\nExamples:\n"
        for i, example in enumerate(examples[:3], 1):  # Limit to 3 examples
            example_text += f"\nExample {i}:\n"
            for key, value in example.items():
                example_text += f"{key}: {value}\n"
                
        return prompt + example_text
        
    def _calculate_optimization_score(
        self,
        original: str,
        optimized: str,
        original_tokens: int,
        optimized_tokens: int
    ) -> float:
        """Calculate optimization score (0-1)"""
        # Factors: token reduction, readability, structure
        token_score = min(1.0, (original_tokens - optimized_tokens) / max(original_tokens, 1))
        readability_score = self._calculate_readability(optimized) / 100
        structure_score = self._calculate_structure_score(optimized)
        
        # Weighted average
        return (token_score * 0.3 + readability_score * 0.4 + structure_score * 0.3)
        
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (Flesch Reading Ease approximation)"""
        sentences = text.split('. ')
        words = text.split()
        syllables = sum([self._count_syllables(word) for word in words])
        
        if len(sentences) == 0 or len(words) == 0:
            return 0
            
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        # Flesch Reading Ease formula
        score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        return max(0, min(100, score))
        
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)"""
        word = word.lower()
        vowels = "aeiouy"
        syllables = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllables += 1
            previous_was_vowel = is_vowel
            
        if word.endswith("e"):
            syllables -= 1
        if syllables == 0:
            syllables = 1
            
        return syllables
        
    def _calculate_structure_score(self, text: str) -> float:
        """Calculate structure score based on formatting"""
        score = 0.5  # Base score
        
        # Check for numbered lists
        if re.search(r'\d+\.', text):
            score += 0.1
            
        # Check for bullet points
        if any(marker in text for marker in ['•', '-', '*']):
            score += 0.1
            
        # Check for sections/headers
        if '\n\n' in text:
            score += 0.1
            
        # Check for clear structure markers
        structure_words = ['first', 'second', 'finally', 'step', 'phase']
        for word in structure_words:
            if word.lower() in text.lower():
                score += 0.05
                
        return min(1.0, score)
        
    def _estimate_cost(self, token_count: int) -> float:
        """Estimate cost based on token count"""
        # GPT-4 pricing: $0.03 per 1K tokens (input)
        return (token_count / 1000) * 0.03
        
    def _get_cache_key(self, template_id: str, variables: Dict[str, Any]) -> str:
        """Generate cache key for prompt"""
        var_str = json.dumps(variables, sort_keys=True)
        return hashlib.md5(f"{template_id}:{var_str}".encode()).hexdigest()
        
    async def test_prompt_variation(
        self,
        template_id: str,
        variable_sets: List[Dict[str, Any]],
        evaluation_criteria: Dict[str, float]
    ) -> Dict[str, Any]:
        """Test prompt variations for performance"""
        results = []
        
        for variables in variable_sets:
            prompt = await self.generate_prompt(template_id, variables)
            
            # Simulate evaluation (in production, this would use actual model output)
            score = np.random.uniform(0.7, 1.0)  # Placeholder
            
            results.append({
                "variables": variables,
                "prompt": prompt[:200] + "...",  # Truncate for display
                "score": score,
                "token_count": len(self.tokenizer.encode(prompt))
            })
            
        # Find best performing variation
        best_result = max(results, key=lambda x: x["score"])
        
        return {
            "template_id": template_id,
            "total_variations": len(variable_sets),
            "results": results,
            "best_variation": best_result,
            "average_score": np.mean([r["score"] for r in results])
        }
        
    async def auto_improve_template(
        self,
        template_id: str,
        performance_data: List[Dict[str, Any]]
    ) -> PromptTemplate:
        """Automatically improve template based on performance data"""
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
            
        template = self.templates[template_id]
        
        # Analyze performance data
        successful_patterns = []
        failed_patterns = []
        
        for data in performance_data:
            if data.get("success", False):
                successful_patterns.append(data.get("prompt_snippet", ""))
            else:
                failed_patterns.append(data.get("prompt_snippet", ""))
                
        # Update template based on patterns
        if successful_patterns:
            # Extract common elements from successful prompts
            common_elements = self._extract_common_elements(successful_patterns)
            
            # Update template to include successful patterns
            template.template = self._incorporate_patterns(template.template, common_elements)
            
        # Update performance metrics
        template.performance_metrics["success_rate"] = len(successful_patterns) / len(performance_data)
        template.performance_metrics["last_updated"] = datetime.utcnow().isoformat()
        
        # Increment version
        version_parts = template.version.split('.')
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        template.version = '.'.join(version_parts)
        
        # Save updated template
        self._save_template(template)
        
        return template
        
    def _extract_common_elements(self, patterns: List[str]) -> List[str]:
        """Extract common elements from successful patterns"""
        if not patterns:
            return []
            
        # Simple word frequency analysis
        word_freq = defaultdict(int)
        for pattern in patterns:
            words = pattern.lower().split()
            for word in words:
                word_freq[word] += 1
                
        # Get most common words (excluding stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        common_words = [
            word for word, freq in word_freq.items()
            if freq > len(patterns) * 0.5 and word not in stop_words
        ]
        
        return common_words[:10]  # Top 10 common elements
        
    def _incorporate_patterns(self, template: str, patterns: List[str]) -> str:
        """Incorporate successful patterns into template"""
        # Add a note about successful patterns
        if patterns:
            pattern_note = f"\n\nKey elements for success: {', '.join(patterns)}"
            if pattern_note not in template:
                template += pattern_note
                
        return template
        
    def get_template_analytics(self) -> Dict[str, Any]:
        """Get analytics for all templates"""
        analytics = {
            "total_templates": len(self.templates),
            "templates_by_type": defaultdict(int),
            "average_performance": {},
            "most_used": [],
            "recently_updated": []
        }
        
        for template_id, template in self.templates.items():
            analytics["templates_by_type"][template.type.value] += 1
            
            if template.performance_metrics:
                analytics["average_performance"][template_id] = template.performance_metrics.get("success_rate", 0)
                
        # Sort by performance
        analytics["top_performing"] = sorted(
            analytics["average_performance"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return analytics
        
    async def generate_video_content_suite(
        self,
        topic: str,
        style: VideoStyle,
        audience: TargetAudience,
        duration: int = 10
    ) -> Dict[str, str]:
        """Generate complete content suite for a video"""
        suite = {}
        
        # Generate script
        script_vars = {
            "topic": topic,
            "style": style.value,
            "audience": audience.value,
            "duration": str(duration),
            "tone": "engaging and informative",
            "num_key_points": "5",
            "context": "Focus on practical value and engagement"
        }
        suite["script"] = await self.generate_prompt("script_gen_v1", script_vars)
        
        # Generate title options
        title_vars = {
            "topic": topic,
            "style": style.value,
            "keywords": topic.lower().replace(" ", ", "),
            "competitor_titles": "N/A"
        }
        suite["titles"] = await self.generate_prompt("title_gen_v1", title_vars)
        
        # Generate hook
        hook_vars = {
            "topic": topic,
            "title": suite["titles"].split('\n')[0] if suite["titles"] else topic,
            "emotion": "curiosity",
            "style": style.value
        }
        suite["hook"] = await self.generate_prompt("hook_gen_v1", hook_vars)
        
        return suite


# Singleton instance
prompt_framework = PromptEngineeringFramework()