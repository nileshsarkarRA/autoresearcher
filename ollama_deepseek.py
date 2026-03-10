"""
Ollama DeepSeek Coder Integration for AutoResearcher.
Integrates DeepSeek Coder LLM via Ollama for autonomous code generation and research.

Usage:
    from ollama_deepseek import OllamaDeepSeek
    
    coder = OllamaDeepSeek(model="deepseek-coder:6.7b-base-q4_0")
    response = coder.generate_code("Generate a function to compute fibonacci numbers")
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from urllib.request import urlopen, Request
from urllib.error import URLError
import time

logger = logging.getLogger(__name__)


class OllamaDeepSeek:
    """Interface to Ollama DeepSeek Coder model for code generation."""
    
    # Supported DeepSeek Coder models optimized for different setups
    AVAILABLE_MODELS = {
        "deepseek-coder:1.3b-base-q4_0": {"ram_required_gb": 2, "speed": "fastest", "quality": "basic"},
        "deepseek-coder:6.7b-base-q4_0": {"ram_required_gb": 8, "speed": "fast", "quality": "good"},
        "deepseek-coder:33b-base-q4_0": {"ram_required_gb": 20, "speed": "medium", "quality": "excellent"},
        "deepseek-coder:base": {"ram_required_gb": 16, "speed": "medium", "quality": "good"},
    }
    
    def __init__(self, 
                 model: str = "deepseek-coder:6.7b-base-q4_0",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 300):
        """
        Initialize Ollama DeepSeek Coder interface.
        
        Args:
            model: Model name (see AVAILABLE_MODELS for options)
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._verify_connection()
    
    def _verify_connection(self) -> bool:
        """Verify connection to Ollama server."""
        try:
            response = urlopen(Request(f"{self.base_url}/api/tags"), timeout=5)
            if response.status == 200:
                logger.info(f"✓ Connected to Ollama at {self.base_url}")
                return True
        except URLError as e:
            logger.error(f"✗ Cannot connect to Ollama at {self.base_url}")
            logger.error(f"  Make sure Ollama is running: ollama serve")
            logger.error(f"  Or pull the model: ollama pull {self.model}")
            raise RuntimeError(f"Ollama connection failed: {e}")
    
    def _pull_model_if_needed(self) -> None:
        """Pull model from Ollama if not already present."""
        try:
            response = urlopen(Request(f"{self.base_url}/api/tags"), timeout=5)
            data = json.loads(response.read().decode())
            installed_models = [m.get("name", "") for m in data.get("models", [])]
            
            if not any(self.model in m for m in installed_models):
                logger.info(f"Model {self.model} not found. Attempting to pull...")
                self._call_ollama({"model": self.model, "stream": False}, endpoint="pull")
        except Exception as e:
            logger.warning(f"Could not check/pull model: {e}")
    
    def _call_ollama(self, 
                     payload: Dict[str, Any], 
                     endpoint: str = "generate") -> str:
        """
        Call Ollama API.
        
        Args:
            payload: Request payload
            endpoint: API endpoint (generate, pull, etc.)
        
        Returns:
            Response text
        """
        url = f"{self.base_url}/api/{endpoint}"
        request = Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        
        try:
            response = urlopen(request, timeout=self.timeout)
            result = ""
            while True:
                chunk = response.read(1024)
                if not chunk:
                    break
                result += chunk.decode()
            return result
        except URLError as e:
            raise RuntimeError(f"Ollama API error: {e}")
    
    def generate_code(self, 
                      prompt: str,
                      language: str = "python",
                      temperature: float = 0.7,
                      top_p: float = 0.95,
                      max_tokens: int = 2048) -> str:
        """
        Generate code using DeepSeek Coder.
        
        Args:
            prompt: Code generation prompt
            language: Programming language (python, javascript, etc.)
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated code
        """
        full_prompt = f"""You are an expert code generator. Generate high-quality {language} code.

Prompt: {prompt}

Generate only the code without explanations or markdown formatting:"""
        
        self._pull_model_if_needed()
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
        }
        
        logger.info(f"Generating {language} code with {self.model}...")
        response = self._call_ollama(payload)
        
        # Parse response (handle both single and streaming formats)
        try:
            data = json.loads(response)
            return data.get("response", "").strip()
        except json.JSONDecodeError:
            # Handle streaming response format
            lines = response.strip().split('\n')
            result = ""
            for line in lines:
                try:
                    data = json.loads(line)
                    result += data.get("response", "")
                except json.JSONDecodeError:
                    continue
            return result.strip()
    
    def optimize_code(self, 
                      code: str,
                      optimization_target: str = "performance") -> str:
        """
        Optimize existing code for A100 GPU.
        
        Args:
            code: Code to optimize
            optimization_target: "performance", "memory", or "both"
        
        Returns:
            Optimized code
        """
        prompt = f"""Optimize this code for {optimization_target} on NVIDIA A100 GPU.
Hardware: A100 80GB, Intel Xeon 42 cores.
Target: Maximize throughput and minimize memory usage.

Original code:
```python
{code}
```

Provide the optimized code:"""
        
        return self.generate_code(prompt, language="python", max_tokens=4096)
    
    def generate_benchmark(self, 
                           code_snippet: str,
                           metrics: list = None) -> str:
        """
        Generate benchmarking code for performance testing.
        
        Args:
            code_snippet: Code to benchmark
            metrics: List of metrics to measure (default: time, memory, throughput)
        
        Returns:
            Benchmarking code
        """
        if metrics is None:
            metrics = ["execution_time", "memory_usage", "throughput"]
        
        prompt = f"""Generate a comprehensive benchmark for this code snippet.
Measure: {', '.join(metrics)}
Hardware: NVIDIA A100 80GB GPU, Intel Xeon 42 cores
Framework: PyTorch with CUDA

Code to benchmark:
```python
{code_snippet}
```

Generate complete benchmarking code with proper setup and reporting:"""
        
        return self.generate_code(prompt, language="python", max_tokens=4096)
    
    def document_code(self, code: str) -> str:
        """
        Generate comprehensive documentation for code.
        
        Args:
            code: Code to document
        
        Returns:
            Documented code with docstrings and comments
        """
        prompt = f"""Add comprehensive documentation to this code.
Include: docstrings, type hints, inline comments, and parameter descriptions.
Target Python 3.12+

Code:
```python
{code}
```

Return fully documented code:"""
        
        return self.generate_code(prompt, language="python", max_tokens=4096)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model."""
        return {
            "model": self.model,
            "specs": self.AVAILABLE_MODELS.get(self.model, {}),
            "connection": self.base_url,
        }


def setup_deepseek_environment():
    """
    Setup instructions for DeepSeek Coder with Ollama.
    """
    setup_guide = """
╔══════════════════════════════════════════════════════════════════╗
║      OLLAMA DEEPSEEK CODER SETUP FOR A100 80GB GPU              ║
╚══════════════════════════════════════════════════════════════════╝

1. INSTALL OLLAMA
   Download from: https://ollama.ai
   Or on Linux: curl https://ollama.ai/install.sh | sh

2. PULL DEEPSEEK CODER MODEL
   For A100 80GB (recommended):
   $ ollama pull deepseek-coder:33b-base-q4_0
   
   For faster iteration (less VRAM):
   $ ollama pull deepseek-coder:6.7b-base-q4_0

3. START OLLAMA SERVER
   $ ollama serve
   
   The server will be available at: http://localhost:11434

4. VERIFY INSTALLATION
   $ curl http://localhost:11434/api/tags
   
   You should see your pulled models in the response.

5. TEST WITH AUTORESEARCHER
   from ollama_deepseek import OllamaDeepSeek
   
   coder = OllamaDeepSeek(model="deepseek-coder:33b-base-q4_0")
   code = coder.generate_code("Generate a vectorized matrix multiplication using PyTorch")
   print(code)

╔══════════════════════════════════════════════════════════════════╗
║                    MODEL RECOMMENDATIONS                         ║
╚══════════════════════════════════════════════════════════════════╝

For NVIDIA A100 80GB GPU:
├─ deepseek-coder:33b-base-q4_0  [RECOMMENDED]
│  ├─ Size: ~20GB VRAM
│  ├─ Quality: Excellent (best for complex code)
│  ├─ Speed: Medium (~2-5 tokens/sec)
│  └─ Use: Production optimization and benchmarking
│
├─ deepseek-coder:6.7b-base-q4_0
│  ├─ Size: ~8GB VRAM
│  ├─ Quality: Good
│  ├─ Speed: Fast (~10-20 tokens/sec)
│  └─ Use: Quick iterations and testing
│
└─ deepseek-coder:1.3b-base-q4_0
   ├─ Size: ~2GB VRAM
   ├─ Quality: Basic
   ├─ Speed: Very fast (~30+ tokens/sec)
   └─ Use: Simple tasks and rapid prototyping

╔══════════════════════════════════════════════════════════════════╗
║                      ENVIRONMENT VARIABLES                       ║
╚══════════════════════════════════════════════════════════════════╝

Optional environment variables:

# Ollama server URL (default: http://localhost:11434)
export OLLAMA_URL=http://localhost:11434

# Number of threads for model execution (default: all cores)
export OLLAMA_NUM_THREADS=42

# GPU memory fraction (default: auto)
export OLLAMA_GPU_MEMORY=0.8

"""
    print(setup_guide)


if __name__ == "__main__":
    # Show setup guide
    setup_deepseek_environment()
    
    # Example usage (if Ollama is running)
    try:
        print("\nTesting Ollama DeepSeek integration...\n")
        coder = OllamaDeepSeek(model="deepseek-coder:6.7b-base-q4_0")
        print(f"✓ Connected to Ollama")
        print(f"✓ Model: {coder.model}")
        
        # Generate a simple example
        code = coder.generate_code(
            "Generate a function to compute the sum of a list",
            max_tokens=256
        )
        print(f"\nGenerated Code:\n{code}\n")
        
    except RuntimeError as e:
        print(f"\n⚠ Ollama not available: {e}\n")
