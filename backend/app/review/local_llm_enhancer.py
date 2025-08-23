"""
Local LLM Enhancer for Code Review Agent
Uses locally downloaded Qwen2.5-7B model through Ollama
100% FREE - No API costs, no rate limits, maximum privacy
"""

import asyncio
import json
import logging
import subprocess
import time
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Structured LLM response"""
    success: bool
    explanation: str
    source: str
    latency_ms: float
    error: Optional[str] = None

class LocalOllamaLLM:
    """Local Ollama LLM integration using Qwen2.5-7B"""
    
    def __init__(self, model_name: str = "qwen2.5-coder:7b"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        self.api_url = f"{self.base_url}/api/generate"
        self.is_available = self._check_availability()
        
        if self.is_available:
            logger.info(f"✅ Local Ollama LLM initialized with {model_name}")
        else:
            logger.warning(f"⚠️ Ollama not available at {self.base_url}")
    
    def _check_availability(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama availability check failed: {e}")
            return False
    
    async def enhance_finding(self, code_snippet: str, ml_finding: Dict) -> LLMResponse:
        """Enhance ML finding using local Qwen2.5-7B model"""
        
        if not self.is_available:
            return LLMResponse(
                success=False,
                explanation="",
                source=f"ollama_local_{self.model_name}",
                latency_ms=0,
                error="Ollama not available - please start with 'ollama serve'"
            )
        
        try:
            start_time = time.time()
            
            # Create optimized prompt for code review
            prompt = self._create_code_review_prompt(code_snippet, ml_finding)
            
            # Make local API request to Ollama
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 512
                }
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120  # Increased timeout for local models
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                explanation = result.get("response", "")
                
                # Clean up the response
                explanation = self._clean_response(explanation, prompt)
                
                return LLMResponse(
                    success=True,
                    explanation=explanation,
                    source=f"ollama_local_{self.model_name}",
                    latency_ms=latency_ms
                )
            else:
                return LLMResponse(
                    success=False,
                    explanation="",
                    source=f"ollama_local_{self.model_name}",
                    latency_ms=latency_ms,
                    error=f"Ollama API error: {response.status_code} - {response.text}"
                )
                
        except Exception as e:
            logger.error(f"Local Ollama error: {e}")
            return LLMResponse(
                success=False,
                explanation="",
                source=f"ollama_local_{self.model_name}",
                latency_ms=0,
                error=str(e)
            )
    
    def _create_code_review_prompt(self, code_snippet: str, ml_finding: Dict) -> str:
        """Create optimized prompt for Qwen2.5-7B code review"""
        
        prompt = f"""<|im_start|>system
You are an expert code reviewer specializing in security, quality, and best practices.
Provide clear, actionable feedback with specific examples and code improvements.
Focus on practical solutions that developers can implement immediately.
<|im_end|>
<|im_start|>user
Analyze this code for issues and provide detailed feedback:

```{ml_finding.get('file', 'unknown')}
{code_snippet}
```

ML Finding: {ml_finding.get('message', 'No message')}
Severity: {ml_finding.get('severity', 'Unknown')}
Category: {ml_finding.get('category', 'Unknown')}

Please provide:
1. **Clear Explanation**: What is the problem in simple terms?
2. **Why It Matters**: Why is this a security/quality/maintainability concern?
3. **How to Fix**: Provide specific, actionable code improvements with examples
4. **Best Practices**: What coding standards should be followed?
5. **Risk Assessment**: How critical is this issue?

Be concise but thorough. Focus on actionable advice.
<|im_end|>
<|im_start|>assistant
"""
        
        return prompt
    
    def _clean_response(self, response: str, prompt: str) -> str:
        """Clean up the LLM response"""
        # Remove the prompt if it's included in the response
        if prompt in response:
            response = response[len(prompt):].strip()
        
        # Remove any system/user/assistant tags
        response = response.replace("<|im_start|>", "").replace("<|im_end|>", "")
        response = response.replace("system", "").replace("user", "").replace("assistant", "")
        
        # Clean up extra whitespace
        response = " ".join(response.split())
        
        return response
    
    async def explain_code_pattern(self, code_snippet: str, question: str) -> str:
        """Answer natural language questions about code"""
        
        # Create a mock finding for the question
        mock_finding = {
            "message": question,
            "severity": "INFO",
            "category": "question",
            "file": "user_query"
        }
        
        response = await self.enhance_finding(code_snippet, mock_finding)
        
        if response.success:
            return response.explanation
        else:
            return f"Error: Could not generate explanation - {response.error}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the local model"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                for model in models:
                    if self.model_name in model.get("name", ""):
                        return {
                            "name": model.get("name"),
                            "size": model.get("size"),
                            "modified_at": model.get("modified_at"),
                            "available": True
                        }
            
            return {"available": False, "error": "Model not found"}
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def restart_service(self) -> bool:
        """Attempt to restart Ollama service"""
        try:
            # Try to start Ollama if it's not running
            subprocess.run(["ollama", "serve"], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL,
                         start_new_session=True)
            
            # Wait a moment for service to start
            time.sleep(3)
            
            # Check if it's now available
            self.is_available = self._check_availability()
            return self.is_available
            
        except Exception as e:
            logger.error(f"Failed to restart Ollama: {e}")
            return False

class LocalLLMEnhancer:
    """Local LLM enhancer with Qwen2.5-7B model"""
    
    def __init__(self, model_name: str = "qwen2.5-coder:7b"):
        self.primary_model = LocalOllamaLLM(model_name)
        self.backup_model = LocalOllamaLLM("qwen2.5-coder:13b") if model_name != "qwen2.5-coder:13b" else None
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.average_latency = 0.0
    
    async def enhance_finding(self, code_snippet: str, ml_finding: Dict) -> Dict[str, Any]:
        """Enhance ML finding with local Qwen2.5-7B model"""
        
        self.total_requests += 1
        
        # Try primary model first
        response = await self.primary_model.enhance_finding(code_snippet, ml_finding)
        
        if response.success:
            self.successful_requests += 1
            self._update_latency_stats(response.latency_ms)
            
            return {
                **ml_finding,
                "llm_enhanced": True,
                "ai_explanation": response.explanation,
                "source": response.source,
                "latency_ms": response.latency_ms,
                "mode": "local_primary"
            }
        
        # Try backup model if available
        if self.backup_model:
            logger.info("🔄 Primary model failed, trying backup model...")
            backup_response = await self.backup_model.enhance_finding(code_snippet, ml_finding)
            
            if backup_response.success:
                self.successful_requests += 1
                self._update_latency_stats(backup_response.latency_ms)
                
                return {
                    **ml_finding,
                    "llm_enhanced": True,
                    "ai_explanation": backup_response.explanation,
                    "source": backup_response.source,
                    "latency_ms": backup_response.latency_ms,
                    "mode": "local_backup"
                }
        
        # All models failed
        self.failed_requests += 1
        logger.error(f"All local LLM models failed: {response.error}")
        
        return {
            **ml_finding,
            "llm_enhanced": False,
            "ai_explanation": f"Error: Local LLM failed - {response.error}",
            "source": "none",
            "latency_ms": 0,
            "mode": "failed"
        }
    
    def _update_latency_stats(self, latency_ms: float):
        """Update average latency statistics"""
        if self.successful_requests == 1:
            self.average_latency = latency_ms
        else:
            self.average_latency = (self.average_latency * (self.successful_requests - 1) + latency_ms) / self.successful_requests
    
    async def explain_code_issue(self, code_snippet: str, question: str) -> str:
        """Use local LLM to explain code issues in natural language"""
        
        if not self.primary_model.is_available:
            return "Local LLM not available - please start Ollama with 'ollama serve'"
        
        return await self.primary_model.explain_code_pattern(code_snippet, question)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate_percent": round(success_rate, 2),
            "average_latency_ms": round(self.average_latency, 2),
            "primary_model": self.primary_model.model_name,
            "backup_model": self.backup_model.model_name if self.backup_model else "none",
            "ollama_available": self.primary_model.is_available
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get comprehensive model status"""
        primary_info = self.primary_model.get_model_info()
        backup_info = self.backup_model.get_model_info() if self.backup_model else {"available": False}
        
        return {
            "primary_model": primary_info,
            "backup_model": backup_info,
            "ollama_service": {
                "running": self.primary_model.is_available,
                "base_url": self.primary_model.base_url
            }
        }
    
    def restart_ollama(self) -> bool:
        """Restart Ollama service"""
        success = self.primary_model.restart_service()
        if success:
            logger.info("✅ Ollama service restarted successfully")
        else:
            logger.warning("⚠️ Failed to restart Ollama service")
        return success
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Local LLM enhancer cleaned up")
        # No specific cleanup needed for local models
