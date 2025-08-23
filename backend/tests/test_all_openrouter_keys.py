#!/usr/bin/env python3
"""
Test all 8 OpenRouter API keys for validity and expiration
"""

import os
import requests
import json
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

def test_api_key(api_key, key_number):
    """Test a single OpenRouter API key"""
    
    if not api_key or api_key.startswith("your_") or api_key.startswith("sk-or-v1-"):
        return False, "Invalid or placeholder key"
    
    # OpenRouter API endpoint
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    # Headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Code Review Agent"
    }
    
    # Simple test payload
    payload = {
        "model": "qwen/qwen2.5-7b-instruct",
        "messages": [
            {
                "role": "user",
                "content": "Hello! Can you help me with code review?"
            }
        ],
        "max_tokens": 50,
        "temperature": 0.3
    }
    
    try:
        print(f"🔑 Testing Key {key_number}: {api_key[:12]}...{api_key[-4:]}")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Key {key_number}: WORKING - API call successful!")
            return True, "Working"
        elif response.status_code == 401:
            print(f"❌ Key {key_number}: EXPIRED/INVALID - Unauthorized")
            return False, "Expired/Invalid"
        elif response.status_code == 429:
            print(f"⚠️ Key {key_number}: RATE LIMITED - Too many requests")
            return False, "Rate Limited"
        elif response.status_code == 402:
            print(f"❌ Key {key_number}: NO CREDITS - Payment required")
            return False, "No Credits"
        else:
            print(f"❌ Key {key_number}: FAILED - Status {response.status_code}")
            print(f"   Response: {response.text[:100]}...")
            return False, f"Failed: {response.status_code}"
            
    except requests.exceptions.Timeout:
        print(f"⏰ Key {key_number}: TIMEOUT - Request took too long")
        return False, "Timeout"
    except requests.exceptions.RequestException as e:
        print(f"❌ Key {key_number}: ERROR - {e}")
        return False, f"Error: {e}"
    except Exception as e:
        print(f"❌ Key {key_number}: UNEXPECTED ERROR - {e}")
        return False, f"Unexpected: {e}"

def test_key_details(api_key, key_number):
    """Get detailed information about a working key"""
    
    if not api_key:
        return
    
    url = "https://openrouter.ai/api/v1/auth/key"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"\n📊 Key {key_number} Details:")
            print(f"   User: {data.get('data', {}).get('user', {}).get('name', 'Unknown')}")
            print(f"   Credits: {data.get('data', {}).get('credits', 'Unknown')}")
            print(f"   Created: {data.get('data', {}).get('created_at', 'Unknown')}")
            print(f"   Last Used: {data.get('data', {}).get('last_used', 'Unknown')}")
        else:
            print(f"⚠️ Could not get details for Key {key_number}")
    except Exception as e:
        print(f"⚠️ Error getting details for Key {key_number}: {e}")

def main():
    """Test all OpenRouter API keys"""
    
    print("🧪 Testing All OpenRouter API Keys")
    print("=" * 60)
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get all 8 keys
    keys = {}
    for i in range(1, 9):
        key = os.getenv(f"OPENROUTER_API_KEY_{i}")
        if key:
            keys[i] = key
    
    if not keys:
        print("❌ No OpenRouter API keys found in .env file!")
        print("💡 Make sure you have OPENROUTER_API_KEY_1 through OPENROUTER_API_KEY_8 set")
        return
    
    print(f"🔑 Found {len(keys)} API keys to test")
    print()
    
    # Test each key
    working_keys = []
    failed_keys = []
    
    for key_number, api_key in keys.items():
        success, status = test_api_key(api_key, key_number)
        
        if success:
            working_keys.append((key_number, api_key))
            # Get detailed info for working keys
            test_key_details(api_key, key_number)
        else:
            failed_keys.append((key_number, api_key, status))
        
        print("-" * 40)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    if working_keys:
        print(f"✅ WORKING KEYS: {len(working_keys)}")
        for key_number, _ in working_keys:
            print(f"   - Key {key_number}")
        
        # Set the first working key as primary
        primary_key_number, primary_key = working_keys[0]
        print(f"\n🎯 PRIMARY KEY: Key {primary_key_number}")
        print(f"💡 Update your .env file with:")
        print(f"   OPENROUTER_API_KEY={primary_key}")
        
    else:
        print("❌ NO WORKING KEYS FOUND")
    
    if failed_keys:
        print(f"\n❌ FAILED KEYS: {len(failed_keys)}")
        for key_number, _, status in failed_keys:
            print(f"   - Key {key_number}: {status}")
    
    print("\n" + "=" * 60)
    
    if working_keys:
        print("🎉 SUCCESS! You have working OpenRouter API keys!")
        print("💡 Your code review agent can now use OpenRouter API")
    else:
        print("⚠️ All keys failed. You may need to:")
        print("   1. Generate new API keys at https://openrouter.ai/keys")
        print("   2. Check if your account has credits")
        print("   3. Verify your account is active")

if __name__ == "__main__":
    main()
