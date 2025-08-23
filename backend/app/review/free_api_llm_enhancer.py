#!/usr/bin/env python3
"""
Free API LLM Enhancer with OpenRouter and Local Fallback
Uses OpenRouter API for free models and falls back to Ollama API
"""

import asyncio
import json
import logging
import time
from enum import Enum
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    """LLM Provider types"""
    OPENROUTER = "openrouter"
    OLLAMA_API = "ollama_api"

class OpenRouterAPI:
    """OpenRouter API integration for free models"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Free OpenRouter models
        self.free_models = [
            "qwen/qwen-2.5-7b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
            "mistralai/mistral-7b-instruct",
            "gpt2"
        ]
    
    async def enhance_finding(self, code: str, finding: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a finding using OpenRouter API"""
        start_time = time.time()
        
        try:
            # Try each free model until one works
            for model in self.free_models:
                try:
                    response = await self._call_openrouter_api(model, code, finding)
                    if response:
                        latency = (time.time() - start_time) * 1000
                        return {
                            **finding,
                            "llm_enhanced": True,
                            "provider": "openrouter",
                            "model": model,
                            "ai_explanation": response,
                            "latency_ms": latency,
                            "llm_suggestion": self._extract_suggestion(response)
                        }
                except Exception as e:
                    logger.warning(f"OpenRouter model {model} failed: {e}")
                    continue
            
            # All models failed
            raise Exception("All OpenRouter models failed")
            
        except Exception as e:
            logger.error(f"OpenRouter API failed: {e}")
            raise
    
    async def _call_openrouter_api(self, model: str, code: str, finding: Dict[str, Any]) -> str:
        """Make API call to OpenRouter"""
        prompt = self._create_code_review_prompt(code, finding)
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")
    
    def _create_code_review_prompt(self, code: str, finding: Dict[str, Any]) -> str:
        """Create a prompt for code review enhancement"""
        return f"""
You are an expert code reviewer. Analyze this code finding and provide:

1. A clear explanation of the issue
2. Why it's important to fix
3. A specific, actionable suggestion to improve the code

Code:
```python
{code}
```

Finding:
- File: {finding.get('file', 'unknown')}
- Line: {finding.get('line', 'unknown')}
- Severity: {finding.get('severity', 'unknown')}
- Category: {finding.get('category', 'unknown')}
- Message: {finding.get('message', 'unknown')}

Provide a professional, concise analysis:
"""
    
    def _extract_suggestion(self, response: str) -> str:
        """Extract suggestion from AI response"""
        # Simple extraction - look for suggestion patterns
        lines = response.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['suggestion:', 'recommend:', 'fix:', 'improve:']):
                return line.strip()
        return response[:100] + "..." if len(response) > 100 else response

class OllamaAPIClient:
    """Ollama API client for local fallback"""
    
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.api_url = f"{self.base_url}/api/generate"
        self.model = "qwen2.5-coder:7b"
        self.is_available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def enhance_finding(self, code: str, finding: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a finding using Ollama API"""
        if not self.is_available:
            raise Exception("Ollama API not available")
        
        start_time = time.time()
        
        try:
            prompt = self._create_code_review_prompt(code, finding)
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "num_predict": 300,
                "temperature": 0.3
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                latency = (time.time() - start_time) * 1000
                return {
                    **finding,
                    "llm_enhanced": True,
                    "provider": "ollama_api",
                    "model": self.model,
                    "ai_explanation": result["response"],
                    "latency_ms": latency,
                    "llm_suggestion": self._extract_suggestion(result["response"])
                }
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Ollama API failed: {e}")
            raise
    
    def _create_code_review_prompt(self, code: str, finding: Dict[str, Any]) -> str:
        """Create a prompt for code review enhancement"""
        return f"""
You are an expert code reviewer. Analyze this code finding and provide:

1. A clear explanation of the issue
2. Why it's important to fix
3. A specific, actionable suggestion to improve the code

Code:
```python
{code}
```

Finding:
- File: {finding.get('file', 'unknown')}
- Line: {finding.get('line', 'unknown')}
- Severity: {finding.get('severity', 'unknown')}
- Category: {finding.get('category', 'unknown')}
- Message: {finding.get('message', 'unknown')}

Provide a professional, concise analysis:
"""
    
    def _extract_suggestion(self, response: str) -> str:
        """Extract suggestion from AI response"""
        lines = response.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['suggestion:', 'recommend:', 'fix:', 'improve:']):
                return line.strip()
        return response[:100] + "..." if len(response) > 100 else response

class FreeAPILLMEnhancer:
    """Free API LLM Enhancer with Local First, OpenRouter Fallback"""
    
    def __init__(self, openrouter_token: str = None):
        self.providers = {}
        self.current_provider = None
        self.provider_failures = {}
        self.max_failures = 3
        
        # Initialize providers with LOCAL FIRST priority
        # Always add Ollama API as PRIMARY (local first)
        self.providers[LLMProvider.OLLAMA_API] = OllamaAPIClient()
        self.current_provider = LLMProvider.OLLAMA_API  # Start with local
        
        # Add OpenRouter as FALLBACK only
        if openrouter_token:
            self.providers[LLMProvider.OPENROUTER] = OpenRouterAPI(openrouter_token)
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.provider_stats = {provider: {"requests": 0, "successes": 0, "failures": 0} for provider in self.providers}
    
    async def enhance_finding(self, code: str, finding: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a finding using LOCAL FIRST, then fallback to OpenRouter"""
        self.total_requests += 1
        
        # ALWAYS try LOCAL LLM first (primary choice)
        if LLMProvider.OLLAMA_API in self.providers and self.provider_failures.get(LLMProvider.OLLAMA_API, 0) < self.max_failures:
            try:
                provider = self.providers[LLMProvider.OLLAMA_API]
                self.provider_stats[LLMProvider.OLLAMA_API]["requests"] += 1
                
                result = await provider.enhance_finding(code, finding)
                
                self.provider_stats[LLMProvider.OLLAMA_API]["successes"] += 1
                self.successful_requests += 1
                self.current_provider = LLMProvider.OLLAMA_API  # Keep local as primary
                
                return result
                
            except Exception as e:
                self.provider_stats[LLMProvider.OLLAMA_API]["failures"] += 1
                self.provider_failures[LLMProvider.OLLAMA_API] = self.provider_failures.get(LLMProvider.OLLAMA_API, 0) + 1
                logger.warning(f"Local LLM failed: {e}")
        
        # Only if LOCAL LLM fails, try OpenRouter as fallback
        if LLMProvider.OPENROUTER in self.providers and self.provider_failures.get(LLMProvider.OPENROUTER, 0) < self.max_failures:
            try:
                logger.info(f"🔄 Local LLM failed, trying OpenRouter as fallback")
                provider = self.providers[LLMProvider.OPENROUTER]
                self.provider_stats[LLMProvider.OPENROUTER]["requests"] += 1
                
                result = await provider.enhance_finding(code, finding)
                
                # Switch to OpenRouter temporarily
                self.current_provider = LLMProvider.OPENROUTER
                self.provider_stats[LLMProvider.OPENROUTER]["successes"] += 1
                self.successful_requests += 1
                
                return result
                
            except Exception as e:
                self.provider_stats[LLMProvider.OPENROUTER]["failures"] += 1
                self.provider_failures[LLMProvider.OPENROUTER] = self.provider_failures.get(LLMProvider.OPENROUTER, 0) + 1
                logger.warning(f"OpenRouter fallback failed: {e}")
        
        # All providers failed
        raise Exception("All LLM providers failed")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": self.successful_requests / self.total_requests if self.total_requests > 0 else 0,
            "provider_stats": self.provider_stats,
            "current_provider": self.current_provider.value if self.current_provider else None
        }
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info(f"📊 Final Free API LLM Stats: {self.total_requests} requests, {self.successful_requests} successful")
