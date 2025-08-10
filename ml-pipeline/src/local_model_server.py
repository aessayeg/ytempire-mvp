"""
Local Llama 2 7B Model Server
Handles local model inference for cost optimization
"""
import os
import torch
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import psutil
import GPUtil
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
    pipeline,
    BitsAndBytesConfig
)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from prometheus_client import Counter, Histogram, Gauge
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for local model"""
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    model_path: str = "./models/llama-2-7b"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 1
    use_8bit: bool = True  # Use 8-bit quantization to save memory
    use_4bit: bool = False  # Use 4-bit quantization for even more savings
    load_in_8bit: bool = True
    torch_dtype: torch.dtype = torch.float16


class LocalModelServer:
    """Local Llama 2 7B model server"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = None
        self.is_loaded = False
        
        # Metrics
        self.inference_counter = Counter(
            'local_model_inference_total',
            'Total number of inference requests'
        )
        self.inference_duration = Histogram(
            'local_model_inference_duration_seconds',
            'Inference duration in seconds'
        )
        self.gpu_memory_usage = Gauge(
            'local_model_gpu_memory_bytes',
            'GPU memory usage in bytes'
        )
        
    def check_gpu_availability(self) -> Dict[str, Any]:
        """Check GPU availability and memory"""
        gpu_info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'devices': []
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                gpu_info['devices'].append({
                    'index': i,
                    'name': device_props.name,
                    'total_memory_gb': device_props.total_memory / (1024**3),
                    'major': device_props.major,
                    'minor': device_props.minor,
                    'multi_processor_count': device_props.multi_processor_count,
                })
                
                # Current memory usage
                if i == 0:  # Check primary GPU
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    gpu_info['current_usage'] = {
                        'allocated_gb': allocated,
                        'reserved_gb': reserved,
                        'free_gb': device_props.total_memory / (1024**3) - reserved
                    }
        
        # Check with GPUtil for more details
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info[f'gpu_{gpu.id}_utilization'] = f"{gpu.load * 100:.1f}%"
                gpu_info[f'gpu_{gpu.id}_memory_used'] = f"{gpu.memoryUsed}MB"
                gpu_info[f'gpu_{gpu.id}_memory_free'] = f"{gpu.memoryFree}MB"
                gpu_info[f'gpu_{gpu.id}_temperature'] = f"{gpu.temperature}°C"
        except Exception as e:
            logger.warning(f"GPUtil check failed: {e}")
        
        return gpu_info
    
    async def load_model(self) -> bool:
        """Load Llama 2 7B model"""
        try:
            logger.info("Starting Llama 2 7B model loading...")
            
            # Check GPU
            gpu_info = self.check_gpu_availability()
            logger.info(f"GPU Info: {json.dumps(gpu_info, indent=2)}")
            
            # Determine device
            if self.config.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
                self.config.load_in_8bit = False
                self.config.use_8bit = False
            else:
                self.device = self.config.device
            
            # Configure quantization
            bnb_config = None
            if self.device == "cuda":
                if self.config.use_4bit:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    logger.info("Using 4-bit quantization")
                elif self.config.use_8bit:
                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.float16
                    )
                    logger.info("Using 8-bit quantization")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path if os.path.exists(self.config.model_path) else self.config.model_name,
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            logger.info("Loading model weights...")
            model_path = self.config.model_path if os.path.exists(self.config.model_path) else self.config.model_name
            
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=self.config.torch_dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to(self.device)
            
            # Create generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                num_beams=self.config.num_beams,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            self.is_loaded = True
            logger.info("Model loaded successfully!")
            
            # Log memory usage
            if self.device == "cuda":
                memory_info = torch.cuda.memory_summary(device=0, abbreviated=True)
                logger.info(f"GPU Memory Summary:\n{memory_info}")
                self.gpu_memory_usage.set(torch.cuda.memory_allocated(0))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            return False
    
    def format_prompt(self, instruction: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt for Llama 2 chat format"""
        if system_prompt:
            return f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{instruction} [/INST]"""
        else:
            return f"<s>[INST] {instruction} [/INST]"
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate text using local model"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            start_time = datetime.utcnow()
            self.inference_counter.inc()
            
            # Format prompt
            formatted_prompt = self.format_prompt(prompt, system_prompt)
            
            # Update generation parameters
            gen_kwargs = {
                'max_new_tokens': max_new_tokens,
                'temperature': temperature or self.config.temperature,
                'top_p': top_p or self.config.top_p,
                'do_sample': True,
                'return_full_text': False
            }
            
            # Generate
            if stream:
                # Streaming generation (for future implementation)
                raise NotImplementedError("Streaming not yet implemented")
            else:
                # Non-streaming generation
                outputs = self.pipeline(
                    formatted_prompt,
                    **gen_kwargs
                )
                
                generated_text = outputs[0]['generated_text']
            
            # Calculate metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.inference_duration.observe(duration)
            
            # Token counting
            input_tokens = len(self.tokenizer.encode(formatted_prompt))
            output_tokens = len(self.tokenizer.encode(generated_text))
            
            # Update GPU memory metric
            if self.device == "cuda":
                self.gpu_memory_usage.set(torch.cuda.memory_allocated(0))
            
            return {
                'generated_text': generated_text,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'generation_time_seconds': duration,
                'tokens_per_second': output_tokens / duration if duration > 0 else 0,
                'model': self.config.model_name,
                'device': self.device,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")
    
    async def generate_script(
        self,
        topic: str,
        style: str = "informative",
        length: str = "medium",
        target_audience: str = "general"
    ) -> Dict[str, Any]:
        """Generate YouTube script using local model"""
        system_prompt = """You are a professional YouTube script writer. 
        Create engaging, well-structured scripts that capture viewer attention.
        Include hooks, clear sections, and calls to action."""
        
        prompt = f"""Write a YouTube video script about: {topic}
        
Style: {style}
Length: {length} (short: 3-5 min, medium: 8-10 min, long: 15+ min)
Target Audience: {target_audience}

Requirements:
1. Start with a strong hook
2. Include clear section breaks
3. Add engagement prompts (like, subscribe)
4. End with a call to action
5. Make it conversational and engaging

Script:"""
        
        max_tokens = {
            'short': 500,
            'medium': 1000,
            'long': 1500
        }.get(length, 1000)
        
        result = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_tokens,
            temperature=0.8
        )
        
        result['script_type'] = 'youtube'
        result['topic'] = topic
        result['style'] = style
        result['length'] = length
        
        return result
    
    def clear_cache(self):
        """Clear GPU cache to free memory"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the model server"""
        health = {
            'status': 'healthy' if self.is_loaded else 'not_loaded',
            'model_loaded': self.is_loaded,
            'device': self.device,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if self.is_loaded:
            # System metrics
            health['system'] = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent
            }
            
            # GPU metrics
            if self.device == "cuda":
                health['gpu'] = {
                    'memory_allocated_gb': torch.cuda.memory_allocated(0) / (1024**3),
                    'memory_reserved_gb': torch.cuda.memory_reserved(0) / (1024**3),
                    'utilization': self.check_gpu_availability()
                }
        
        return health


# FastAPI app for serving the model
app = FastAPI(title="Local Llama 2 Model Server")
model_server = LocalModelServer()


class GenerationRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


class ScriptRequest(BaseModel):
    topic: str
    style: str = "informative"
    length: str = "medium"
    target_audience: str = "general"


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    await model_server.load_model()


@app.get("/health")
async def health():
    """Health check endpoint"""
    return await model_server.health_check()


@app.post("/generate")
async def generate(request: GenerationRequest):
    """Generate text endpoint"""
    try:
        result = await model_server.generate(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_script")
async def generate_script(request: ScriptRequest):
    """Generate YouTube script endpoint"""
    try:
        result = await model_server.generate_script(
            topic=request.topic,
            style=request.style,
            length=request.length,
            target_audience=request.target_audience
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gpu_info")
async def gpu_info():
    """Get GPU information"""
    return model_server.check_gpu_availability()


@app.post("/clear_cache")
async def clear_cache():
    """Clear GPU cache"""
    model_server.clear_cache()
    return {"status": "cache_cleared"}


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )