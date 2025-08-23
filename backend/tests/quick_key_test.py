#!/usr/bin/env python3
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_key(key_num, api_key):
    print(f"🔑 Testing Key {key_num}: {api_key[:12]}...{api_key[-4:]}")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Code Review Agent"
    }
    
    payload = {
        "model": "qwen/qwen2.5-7b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            print(f"✅ Key {key_num}: WORKING!")
            return True
        elif response.status_code == 401:
            print(f"❌ Key {key_num}: EXPIRED/INVALID")
            return False
        elif response.status_code == 402:
            print(f"❌ Key {key_num}: NO CREDITS")
            return False
        else:
            print(f"❌ Key {key_num}: FAILED ({response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Key {key_num}: ERROR - {e}")
        return False

print("🧪 Quick OpenRouter API Key Test")
print("=" * 50)

working_keys = []
for i in range(1, 9):
    key = os.getenv(f"OPENROUTER_API_KEY_{i}")
    if key:
        if test_key(i, key):
            working_keys.append((i, key))
    print("-" * 30)

print(f"\n📊 RESULTS: {len(working_keys)} working keys out of 8")
if working_keys:
    print("✅ Working keys:")
    for num, key in working_keys:
        print(f"   Key {num}: {key[:12]}...{key[-4:]}")
    
    # Set first working key as primary
    primary_num, primary_key = working_keys[0]
    print(f"\n🎯 Use Key {primary_num} as your primary key")
    print(f"💡 Add to .env: OPENROUTER_API_KEY={primary_key}")
else:
    print("❌ No working keys found")
