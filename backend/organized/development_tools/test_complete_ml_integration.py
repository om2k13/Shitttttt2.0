#!/usr/bin/env python3
"""
Test Complete ML + Neural Network Integration
Tests the entire pipeline with all trained models
"""

import asyncio
import json
from pathlib import Path
import sys
import os

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.review.code_review_agent import CodeReviewAgent
from app.review.production_ml_analyzer import ProductionMLAnalyzer, get_ml_analyzer

def create_test_files():
    """Create test files for analysis"""
    test_dir = Path("test_code_samples")
    test_dir.mkdir(exist_ok=True)
    
    # Test Python file with potential security issues
    python_test = test_dir / "vulnerable_code.py"
    python_test.write_text("""
import os
import subprocess
from flask import Flask, request

app = Flask(__name__)

# Hardcoded credentials (security issue)
DATABASE_PASSWORD = "admin123"
API_KEY = "sk-1234567890abcdef"

@app.route('/execute', methods=['POST'])
def execute_command():
    # Command injection vulnerability
    command = request.form.get('command')
    result = os.system(command)  # Dangerous!
    return f"Command executed: {result}"

@app.route('/search')
def search_users():
    # SQL injection vulnerability
    query = request.args.get('query')
    sql = f"SELECT * FROM users WHERE name = '{query}'"  # Vulnerable!
    # Execute SQL query...
    return "Search results"

def complex_function_with_issues(data, options, flags, config, params):
    # High complexity and nesting
    if data:
        if options:
            if flags:
                if config:
                    if params:
                        for item in data:
                            if item.get('active'):
                                if item.get('validated'):
                                    if item.get('processed'):
                                        # Deep nesting continues...
                                        result = process_item(item, options, flags)
                                        if result:
                                            return result
    return None

def process_item(item, options, flags):
    # More complexity
    total = 0
    for i in range(100):
        if i % 2 == 0:
            if options.get('double'):
                total += i * 2
            else:
                total += i
        else:
            if flags.get('triple'):
                total += i * 3
            else:
                total += i
    return total

# Eval usage (dangerous)
def dynamic_execution(code_string):
    return eval(code_string)  # Security risk!

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')  # Debug mode in production!
""")
    
    # Test JavaScript file with issues
    js_test = test_dir / "risky_frontend.js"
    js_test.write_text("""
// Vulnerable JavaScript code
function authenticateUser(username, password) {
    // Hardcoded credentials
    const ADMIN_PASSWORD = "admin123";
    
    if (password === ADMIN_PASSWORD) {
        return true;
    }
    
    // XSS vulnerability
    document.getElementById('welcome').innerHTML = "Welcome " + username;
    
    return false;
}

function processUserInput() {
    const userInput = document.getElementById('userInput').value;
    
    // Dangerous eval usage
    try {
        const result = eval(userInput);
        document.getElementById('result').innerHTML = result;
    } catch (e) {
        console.error('Error:', e);
    }
}

function complexNestedFunction(data, options, config) {
    if (data) {
        if (data.length > 0) {
            if (options) {
                if (options.enabled) {
                    if (config) {
                        if (config.advanced) {
                            for (let i = 0; i < data.length; i++) {
                                if (data[i].active) {
                                    if (data[i].validated) {
                                        if (data[i].processed) {
                                            // Deep nesting...
                                            const result = processDataItem(data[i], options, config);
                                            if (result && result.success) {
                                                return result.data;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return null;
}

// AJAX call without proper error handling
function makeApiCall(endpoint, data) {
    fetch(endpoint, {
        method: 'POST',
        body: JSON.stringify(data),
        headers: {
            'Content-Type': 'application/json'
        }
    }).then(response => response.json())
      .then(data => {
          // Direct DOM manipulation without sanitization
          document.body.innerHTML += data.html;
      });
}

// Timeout with string code (security risk)
setTimeout("alert('This is dangerous')", 1000);

export { authenticateUser, processUserInput, complexNestedFunction };
""")
    
    # Test good quality code for comparison
    good_code = test_dir / "good_quality.py"
    good_code.write_text("""# Well-structured, secure Python code example
import hashlib
import secrets
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class UserCredentials:
    username: str
    password_hash: str
    salt: str

class SecureUserManager:
    def __init__(self):
        self.users: Dict[str, UserCredentials] = {}
    
    def create_user(self, username: str, password: str) -> bool:
        \"\"\"Create a new user with secure password hashing\"\"\"
        if self.user_exists(username):
            return False
        
        salt = secrets.token_hex(32)
        password_hash = self._hash_password(password, salt)
        
        self.users[username] = UserCredentials(
            username=username,
            password_hash=password_hash,
            salt=salt
        )
        
        logger.info(f"User created: {username}")
        return True
    
    def authenticate_user(self, username: str, password: str) -> bool:
        \"\"\"Authenticate user with secure password verification\"\"\"
        if not self.user_exists(username):
            return False
        
        user = self.users[username]
        expected_hash = self._hash_password(password, user.salt)
        
        return secrets.compare_digest(expected_hash, user.password_hash)
    
    def user_exists(self, username: str) -> bool:
        \"\"\"Check if user exists\"\"\"
        return username in self.users
    
    def _hash_password(self, password: str, salt: str) -> str:
        \"\"\"Hash password with salt using secure algorithm\"\"\"
        return hashlib.pbkdf2_hmac('sha256', 
                                   password.encode('utf-8'), 
                                   salt.encode('utf-8'), 
                                   100000).hex()

def validate_input(user_input: str) -> Optional[str]:
    \"\"\"Validate and sanitize user input\"\"\"
    if not user_input or len(user_input) > 1000:
        return None
    
    # Remove potentially dangerous characters
    sanitized = ''.join(c for c in user_input if c.isalnum() or c in ' .-_')
    return sanitized.strip()

def process_data_safely(data: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Process data with proper error handling\"\"\"
    try:
        result = {
            'processed': True,
            'timestamp': secrets.token_hex(16),
            'data_length': len(str(data))
        }
        return result
    
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return {'processed': False, 'error': 'Processing failed'}
""")
    
    return test_dir

async def test_production_ml_analyzer():
    """Test the production ML analyzer directly"""
    print("🧪 Testing Production ML Analyzer...")
    
    try:
        analyzer = get_ml_analyzer()
        status = analyzer.get_model_status()
        
        print(f"  ✅ Models loaded: {status['total_models']}")
        print(f"  📊 Traditional ML: {status['traditional_models_loaded']}")
        print(f"  🧠 Neural Networks: {status['neural_models_loaded']}")
        print(f"  🔧 Scaler loaded: {status['scaler_loaded']}")
        print(f"  🎯 Ensemble loaded: {status['ensemble_loaded']}")
        
        # Test code analysis
        test_code = """
import os
import subprocess

def vulnerable_function(user_input):
    # This is dangerous
    result = os.system(user_input)
    return result

def complex_nested_function(data):
    if data:
        if data.get('active'):
            if data.get('validated'):
                if data.get('processed'):
                    for item in data.get('items', []):
                        if item.get('enabled'):
                            # Deep nesting...
                            return process_item(item)
    return None
"""
        
        print("  🔍 Analyzing test code...")
        results = analyzer.analyze_code_ml(test_code, "test.py")
        
        print(f"  📊 Features extracted: {len(results.get('features', {}))}")
        print(f"  🎯 Risk level: {results.get('risk_assessment', {}).get('risk_level', 'UNKNOWN')}")
        print(f"  🔢 Risk score: {results.get('risk_assessment', {}).get('risk_score', 0.0):.2f}")
        print(f"  📝 Recommendations: {len(results.get('recommendations', []))}")
        
        if results.get('ensemble_prediction'):
            ensemble = results['ensemble_prediction']
            print(f"  🎯 Ensemble vulnerability: {ensemble.get('is_vulnerable', False)}")
            print(f"  🎯 Ensemble confidence: {ensemble.get('confidence', 0.0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing ML analyzer: {e}")
        return False

async def test_code_review_agent_integration():
    """Test the complete code review agent with ML integration"""
    print("🚀 Testing Complete Code Review Agent Integration...")
    
    try:
        # Create test files
        test_dir = create_test_files()
        
        # Initialize code review agent
        agent = CodeReviewAgent(repo_path=test_dir)
        
        # Check if production ML analyzer is loaded
        if agent.production_ml_analyzer:
            print("  ✅ Production ML analyzer loaded in agent")
            status = agent.production_ml_analyzer.get_model_status()
            print(f"    📊 Total models: {status['total_models']}")
        else:
            print("  ❌ Production ML analyzer not loaded")
            return False
        
        # Run complete code review
        print("  🔍 Running complete code review with ML analysis...")
        results = await agent.run_code_review()
        
        # Analyze results
        if results.get('status') == 'success':
            findings = results.get('findings', [])
            print(f"  ✅ Code review completed successfully")
            print(f"  📊 Total findings: {len(findings)}")
            
            # Count findings by category
            categories = {}
            ml_findings = 0
            
            for finding in findings:
                category = finding.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
                
                if 'ml_' in category or 'neural_' in category:
                    ml_findings += 1
            
            print(f"  🤖 ML/Neural findings: {ml_findings}")
            print("  📋 Findings by category:")
            for cat, count in sorted(categories.items()):
                print(f"    • {cat}: {count}")
            
            # Show some ML findings
            ml_findings_list = [f for f in findings if 'ml_' in f.get('category', '') or 'neural_' in f.get('category', '')]
            if ml_findings_list:
                print("  🎯 Sample ML findings:")
                for finding in ml_findings_list[:3]:
                    print(f"    • {finding.get('category', 'unknown')}: {finding.get('message', 'No message')[:80]}...")
            
            return True
        else:
            print(f"  ❌ Code review failed: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error testing integration: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_model_predictions():
    """Test specific model predictions"""
    print("🎯 Testing Individual Model Predictions...")
    
    try:
        analyzer = get_ml_analyzer()
        
        # Test vulnerable code
        vulnerable_code = """
import os
import subprocess

def execute_user_command(command):
    return os.system(command)  # Very dangerous!

def sql_query(user_input):
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    return query  # SQL injection risk

password = "hardcoded_password"  # Bad practice
"""
        
        # Test safe code
        safe_code = """
import hashlib
import secrets
from typing import Optional

def hash_password(password: str, salt: Optional[str] = None) -> str:
    if salt is None:
        salt = secrets.token_hex(32)
    
    return hashlib.pbkdf2_hmac('sha256', 
                              password.encode('utf-8'), 
                              salt.encode('utf-8'), 
                              100000).hex()

def validate_input(user_input: str) -> bool:
    return len(user_input) <= 100 and user_input.isalnum()
"""
        
        print("  🔍 Testing vulnerable code...")
        vuln_results = analyzer.analyze_code_ml(vulnerable_code, "vulnerable.py")
        vuln_risk = vuln_results.get('risk_assessment', {})
        print(f"    Risk level: {vuln_risk.get('risk_level', 'UNKNOWN')}")
        print(f"    Risk score: {vuln_risk.get('risk_score', 0.0):.2f}")
        
        if vuln_results.get('ensemble_prediction'):
            ensemble = vuln_results['ensemble_prediction']
            print(f"    Ensemble vulnerable: {ensemble.get('is_vulnerable', False)}")
        
        print("  🔍 Testing safe code...")
        safe_results = analyzer.analyze_code_ml(safe_code, "safe.py")
        safe_risk = safe_results.get('risk_assessment', {})
        print(f"    Risk level: {safe_risk.get('risk_level', 'UNKNOWN')}")
        print(f"    Risk score: {safe_risk.get('risk_score', 0.0):.2f}")
        
        if safe_results.get('ensemble_prediction'):
            ensemble = safe_results['ensemble_prediction']
            print(f"    Ensemble vulnerable: {ensemble.get('is_vulnerable', False)}")
        
        # Compare predictions
        vuln_score = vuln_risk.get('risk_score', 0.0)
        safe_score = safe_risk.get('risk_score', 0.0)
        
        if vuln_score > safe_score:
            print(f"  ✅ ML correctly identified vulnerable code as riskier ({vuln_score:.2f} vs {safe_score:.2f})")
            return True
        else:
            print(f"  ⚠️ Unexpected: Safe code scored higher ({safe_score:.2f} vs {vuln_score:.2f})")
            return False
            
    except Exception as e:
        print(f"  ❌ Error testing predictions: {e}")
        return False

async def cleanup_test_files():
    """Clean up test files"""
    try:
        test_dir = Path("test_code_samples")
        if test_dir.exists():
            for file in test_dir.iterdir():
                file.unlink()
            test_dir.rmdir()
        print("🧹 Cleaned up test files")
    except Exception as e:
        print(f"Warning: Could not clean up test files: {e}")

async def main():
    """Run all tests"""
    print("🧪 COMPREHENSIVE ML + NEURAL NETWORK INTEGRATION TEST")
    print("=" * 80)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Production ML Analyzer
    if await test_production_ml_analyzer():
        tests_passed += 1
        print("✅ Test 1: Production ML Analyzer - PASSED")
    else:
        print("❌ Test 1: Production ML Analyzer - FAILED")
    
    print()
    
    # Test 2: Code Review Agent Integration
    if await test_code_review_agent_integration():
        tests_passed += 1
        print("✅ Test 2: Code Review Agent Integration - PASSED")
    else:
        print("❌ Test 2: Code Review Agent Integration - FAILED")
    
    print()
    
    # Test 3: Model Predictions
    if await test_model_predictions():
        tests_passed += 1
        print("✅ Test 3: Model Predictions - PASSED")
    else:
        print("❌ Test 3: Model Predictions - FAILED")
    
    print()
    print("=" * 80)
    print(f"🎯 INTEGRATION TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! ML + Neural Network integration is working!")
        print("🚀 Production-ready ML-enhanced code review system is live!")
    else:
        print("⚠️ Some tests failed. Please review the errors above.")
    
    # Cleanup
    await cleanup_test_files()
    
    return tests_passed == total_tests

if __name__ == "__main__":
    asyncio.run(main())
