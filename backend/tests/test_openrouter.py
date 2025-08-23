#!/usr/bin/env python3
"""
Test OpenRouter API integration with Qwen2.5-7B-Instruct
"""

import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from current directory
load_dotenv(".env")

def test_openrouter_api():
    """Test OpenRouter API with Qwen2.5-7B-Instruct"""
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key or api_key == "your_openrouter_key_here":
        print("❌ Please set your OPENROUTER_API_KEY in .env file")
        print("💡 Get it from: https://openrouter.ai/keys")
        return False
    
    # OpenRouter API endpoint
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    # Headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",  # Your app URL
        "X-Title": "Code Review Agent"  # Your app name
    }
    
    # Test prompt for code review
    test_code = """
def calculate_factorial(n):
    if n < 0:
        return None
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
    """
    
    # Payload for Qwen2.5-7B-Instruct
    payload = {
        "model": "qwen/qwen-2.5-7b-instruct",  # OpenRouter model ID
        "messages": [
            {
                "role": "system",
                "content": "You are an expert code reviewer. Analyze the given code for quality, security, and best practices."
            },
            {
                "role": "user", 
                "content": f"Please review this Python code and provide feedback:\n\n{test_code}"
            }
        ],
        "max_tokens": 500,
        "temperature": 0.3
    }
    
    try:
        print("🚀 Testing OpenRouter API with Qwen2.5-7B-Instruct...")
        print(f"📝 Model: qwen/qwen-2.5-7b-instruct")
        print(f"🔑 API Key: {api_key[:8]}...{api_key[-4:]}")
        print("⏳ Sending request...")
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call successful!")
            print(f"📊 Response: {json.dumps(result, indent=2)}")
            
            # Extract the response content
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                print(f"\n🤖 AI Response:\n{content}")
            else:
                print("⚠️ No content in response")
                
            return True
        else:
            print(f"❌ API call failed with status {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_alternative_models():
    """Test alternative free models if Qwen fails"""
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_openrouter_key_here":
        return
    
    # Alternative free models
    models = [
        "meta-llama/codellama-2-7b-instruct",  # CodeLlama
        "gpt2",  # GPT-2
        "microsoft/dialo-gpt-small"  # DialoGPT
    ]
    
    print("\n🔄 Testing alternative free models...")
    
    for model in models:
        print(f"\n📝 Testing model: {model}")
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Code Review Agent"
        }
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello! Can you help me with code review?"
                }
            ],
            "max_tokens": 100,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                print(f"✅ {model}: Working")
            else:
                print(f"❌ {model}: Failed ({response.status_code})")
        except Exception as e:
            print(f"❌ {model}: Error - {e}")

if __name__ == "__main__":
    print("🧪 OpenRouter API Integration Test")
    print("=" * 50)
    
    # Test main model
    success = test_openrouter_api()
    
    # Test alternatives if main fails
    if not success:
        test_alternative_models()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 OpenRouter integration successful!")
        print("💡 You can now use this in your code review agent")
    else:
        print("⚠️ OpenRouter integration needs configuration")
        print("🔑 Make sure to set your API key in .env file")
