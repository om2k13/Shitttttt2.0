"""
Create Comprehensive Training Data for Code Review ML Models

This script combines:
1. Our existing real code analysis data
2. Industry-standard security patterns (OWASP, Bandit, SonarQube)
3. Synthetic data based on industry best practices
4. Real vulnerability patterns from our existing datasets
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

class ComprehensiveTrainingDataCreator:
    """Creates comprehensive training data combining multiple sources"""
    
    def __init__(self, output_dir: str = "comprehensive_training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_existing_data(self):
        """Load our existing real code analysis data"""
        print("📊 Loading existing real code analysis data...")
        
        existing_data = []
        
        # Load real code analysis
        real_analysis_file = Path("real_code_analysis/features.npy")
        if real_analysis_file.exists():
            features = np.load(real_analysis_file)
            quality_labels = np.load("real_code_analysis/quality_labels.npy")
            security_labels = np.load("real_code_analysis/security_labels.npy")
            
            for i, feature_vector in enumerate(features):
                existing_data.append({
                    'features': feature_vector.tolist(),
                    'quality_label': int(quality_labels[i]),
                    'security_label': int(security_labels[i]),
                    'source': 'real_code_analysis',
                    'data_type': 'real'
                })
            
            print(f"✅ Loaded {len(existing_data)} real code analysis samples")
        
        return existing_data
    
    def create_industry_security_patterns(self):
        """Create industry-standard security vulnerability patterns"""
        print("🛡️ Creating industry security patterns...")
        
        security_patterns = [
            # OWASP Top 10 patterns
            {
                'pattern': 'sql_injection',
                'examples': [
                    "query = f\"SELECT * FROM users WHERE id = {user_input}\"",
                    "query = \"SELECT * FROM users WHERE username='\" + username + \"'\"",
                    "cursor.execute(\"SELECT * FROM users WHERE id = \" + user_id)"
                ],
                'severity': 'critical',
                'cwe': 'CWE-89',
                'description': 'SQL injection via string concatenation',
                'remediation': 'Use parameterized queries or ORM'
            },
            {
                'pattern': 'xss_vulnerability',
                'examples': [
                    "innerHTML = user_input",
                    "document.write(user_input)",
                    "element.innerHTML = user_input"
                ],
                'severity': 'high',
                'cwe': 'CWE-79',
                'description': 'Cross-site scripting via innerHTML',
                'remediation': 'Use textContent or proper escaping'
            },
            {
                'pattern': 'command_injection',
                'examples': [
                    "os.system(user_input)",
                    "subprocess.call(user_input, shell=True)",
                    "subprocess.Popen(user_input, shell=True)"
                ],
                'severity': 'critical',
                'cwe': 'CWE-78',
                'description': 'Command injection via os.system',
                'remediation': 'Use subprocess with shell=False'
            },
            {
                'pattern': 'path_traversal',
                'examples': [
                    "open(user_input)",
                    "with open(user_input, 'r') as f:",
                    "file_path = user_input"
                ],
                'severity': 'high',
                'cwe': 'CWE-22',
                'description': 'Path traversal vulnerability',
                'remediation': 'Validate and sanitize file paths'
            },
            {
                'pattern': 'deserialization_vulnerability',
                'examples': [
                    "pickle.loads(user_input)",
                    "yaml.load(user_input)",
                    "json.loads(user_input)"
                ],
                'severity': 'high',
                'cwe': 'CWE-502',
                'description': 'Insecure deserialization',
                'remediation': 'Use safe deserialization methods'
            },
            {
                'pattern': 'hardcoded_credentials',
                'examples': [
                    "password = 'admin123'",
                    "api_key = 'sk-1234567890abcdef'",
                    "secret = 'mysecretkey'"
                ],
                'severity': 'medium',
                'cwe': 'CWE-259',
                'description': 'Hardcoded credentials',
                'remediation': 'Use environment variables or secure vaults'
            },
            {
                'pattern': 'weak_cryptography',
                'examples': [
                    "hashlib.md5(password)",
                    "hashlib.sha1(password)",
                    "random.randint(1, 100)"
                ],
                'severity': 'medium',
                'cwe': 'CWE-327',
                'description': 'Weak cryptographic algorithms',
                'remediation': 'Use strong algorithms like SHA-256, bcrypt'
            }
        ]
        
        return security_patterns
    
    def create_industry_quality_patterns(self):
        """Create industry-standard code quality patterns"""
        print("🔍 Creating industry quality patterns...")
        
        quality_patterns = [
            # SonarQube-style patterns
            {
                'pattern': 'long_function',
                'threshold': 20,
                'severity': 'medium',
                'description': 'Function exceeds 20 lines',
                'impact': 'Reduced readability and maintainability'
            },
            {
                'pattern': 'high_complexity',
                'threshold': 10,
                'severity': 'major',
                'description': 'Cyclomatic complexity exceeds 10',
                'impact': 'Hard to understand and test'
            },
            {
                'pattern': 'deep_nesting',
                'threshold': 4,
                'severity': 'medium',
                'description': 'Nesting depth exceeds 4 levels',
                'impact': 'Code becomes hard to follow'
            },
            {
                'pattern': 'many_parameters',
                'threshold': 5,
                'severity': 'major',
                'description': 'Function has more than 5 parameters',
                'impact': 'Hard to use and maintain'
            },
            {
                'pattern': 'unused_variables',
                'severity': 'minor',
                'description': 'Unused local variables',
                'impact': 'Code clutter and confusion'
            },
            {
                'pattern': 'magic_numbers',
                'severity': 'minor',
                'description': 'Magic numbers in code',
                'impact': 'Unclear business logic'
            },
            {
                'pattern': 'duplicate_code',
                'severity': 'major',
                'description': 'Code duplication detected',
                'impact': 'Maintenance overhead and bugs'
            }
        ]
        
        return quality_patterns
    
    def generate_comprehensive_training_data(self, base_samples: int = 5000):
        """Generate comprehensive training data combining all sources"""
        print(f"🏭 Generating {base_samples} comprehensive training samples...")
        
        training_samples = []
        
        # Load existing data
        existing_data = self.load_existing_data()
        
        # Ensure existing data has consistent feature length (pad with zeros if needed)
        for sample in existing_data:
            if len(sample['features']) == 6:  # Our existing data has 6 features
                # Pad with 3 additional features for security analysis
                sample['features'].extend([0, 0, 0])  # Default security features
            
            # Convert numpy types to Python types for JSON serialization
            sample['features'] = [int(x) for x in sample['features']]
            sample['quality_label'] = int(sample['quality_label'])
            sample['security_label'] = int(sample['security_label'])
        
        training_samples.extend(existing_data)
        
        # Generate additional synthetic data based on industry patterns
        for i in range(base_samples - len(existing_data)):
            # Generate realistic code metrics
            lines = np.random.randint(5, 300)
            complexity = np.random.randint(1, max(2, min(50, lines // 2)))
            nesting = np.random.randint(1, max(2, min(12, complexity // 2)))
            imports = np.random.randint(0, max(1, min(100, lines // 3)))
            functions = np.random.randint(1, max(2, min(40, lines // 5)))
            classes = np.random.randint(0, max(1, min(20, lines // 10)))
            
            # Base feature vector
            features = [lines, complexity, nesting, imports, functions, classes]
            
            # Determine security risk based on industry patterns
            security_risk = 0
            security_score = 0.0
            
            # High complexity + deep nesting = security risk
            if complexity > 20 and nesting > 6:
                security_risk = 1
                security_score = np.random.uniform(0.8, 1.0)
            elif complexity > 15 and nesting > 4:
                security_risk = np.random.choice([0, 1], p=[0.3, 0.7])
                security_score = np.random.uniform(0.6, 0.9) if security_risk else np.random.uniform(0.1, 0.4)
            elif complexity > 10 and nesting > 3:
                security_risk = np.random.choice([0, 1], p=[0.6, 0.4])
                security_score = np.random.uniform(0.4, 0.8) if security_risk else np.random.uniform(0.0, 0.3)
            else:
                security_risk = np.random.choice([0, 1], p=[0.9, 0.1])
                security_score = np.random.uniform(0.7, 1.0) if security_risk else np.random.uniform(0.0, 0.2)
            
            # Add security-related features (always 3 additional features)
            if security_risk:
                features.extend([
                    np.random.randint(6, 10),  # High security risk
                    np.random.randint(3, 8),   # Multiple user inputs
                    np.random.randint(2, 6)    # Multiple external calls
                ])
            else:
                features.extend([
                    np.random.randint(0, 3),   # Low security risk
                    np.random.randint(0, 2),   # Few user inputs
                    np.random.randint(0, 1)    # Few external calls
                ])
            
            # Calculate quality score based on industry standards
            quality_score = 1.0
            
            # Penalize based on SonarQube-style rules
            if lines > 100:
                quality_score -= 0.2
            if lines > 200:
                quality_score -= 0.3
                
            if complexity > 15:
                quality_score -= 0.3
            if complexity > 25:
                quality_score -= 0.4
                
            if nesting > 4:
                quality_score -= 0.2
            if nesting > 6:
                quality_score -= 0.3
                
            if functions > 15:
                quality_score -= 0.1
            if functions > 25:
                quality_score -= 0.2
                
            if classes > 10:
                quality_score -= 0.1
            if classes > 15:
                quality_score -= 0.2
            
            quality_score = max(0.0, quality_score)
            
            # Convert to quality label (0=high, 1=medium, 2=low)
            if quality_score >= 0.8:
                quality_label = 0
            elif quality_score >= 0.6:
                quality_label = 1
            else:
                quality_label = 2
            
            training_samples.append({
                'features': [int(x) for x in features],  # Convert to Python ints
                'quality_label': int(quality_label),
                'security_label': int(security_risk),
                'quality_score': float(quality_score),
                'security_score': float(security_score),
                'source': 'synthetic_industry_based',
                'data_type': 'synthetic',
                'metadata': {
                    'lines': int(lines),
                    'complexity': int(complexity),
                    'nesting': int(nesting),
                    'imports': int(imports),
                    'functions': int(functions),
                    'classes': int(classes)
                }
            })
        
        return training_samples
    
    def create_ml_ready_dataset(self, training_samples: List[Dict]):
        """Convert training samples to ML-ready format"""
        print("🔧 Converting to ML-ready format...")
        
        # Extract features and labels
        X = []
        y_quality = []
        y_security = []
        
        for sample in training_samples:
            X.append(sample['features'])
            y_quality.append(sample['quality_label'])
            y_security.append(sample['security_label'])
        
        # Convert to numpy arrays
        X = np.array(X)
        y_quality = np.array(y_quality)
        y_security = np.array(y_security)
        
        return X, y_quality, y_security
    
    def save_comprehensive_dataset(self, training_samples: List[Dict], X: np.ndarray, y_quality: np.ndarray, y_security: np.ndarray):
        """Save the comprehensive dataset"""
        print("💾 Saving comprehensive dataset...")
        
        # Save detailed training samples
        samples_file = self.output_dir / "comprehensive_training_samples.json"
        with open(samples_file, 'w') as f:
            json.dump(training_samples, f, indent=2)
        
        # Save ML-ready arrays
        np.save(self.output_dir / "comprehensive_features.npy", X)
        np.save(self.output_dir / "comprehensive_quality_labels.npy", y_quality)
        np.save(self.output_dir / "comprehensive_security_labels.npy", y_security)
        
        # Save industry patterns
        security_patterns = self.create_industry_security_patterns()
        quality_patterns = self.create_industry_quality_patterns()
        
        patterns_file = self.output_dir / "industry_patterns.json"
        with open(patterns_file, 'w') as f:
            json.dump({
                'security_patterns': security_patterns,
                'quality_patterns': quality_patterns
            }, f, indent=2)
        
        # Save metadata
        metadata = {
            'total_samples': len(training_samples),
            'real_samples': len([s for s in training_samples if s['data_type'] == 'real']),
            'synthetic_samples': len([s for s in training_samples if s['data_type'] == 'synthetic']),
            'features': X.shape[1],
            'quality_distribution': {
                'high': int(np.sum(y_quality == 0)),
                'medium': int(np.sum(y_quality == 1)),
                'low': int(np.sum(y_quality == 2))
            },
            'security_distribution': {
                'safe': int(np.sum(y_security == 0)),
                'risky': int(np.sum(y_security == 1))
            },
            'data_sources': list(set(s['source'] for s in training_samples)),
            'industry_patterns': {
                'security': len(security_patterns),
                'quality': len(quality_patterns)
            }
        }
        
        metadata_file = self.output_dir / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Comprehensive dataset saved to {self.output_dir}")
        print(f"📊 Total samples: {metadata['total_samples']}")
        print(f"📊 Real samples: {metadata['real_samples']}")
        print(f"📊 Synthetic samples: {metadata['synthetic_samples']}")
        print(f"📊 Features: {metadata['features']}")
        print(f"📊 Industry patterns: {metadata['industry_patterns']}")
        
        return self.output_dir
    
    def create_all(self, base_samples: int = 5000):
        """Create the complete comprehensive training dataset"""
        print("🚀 Creating Comprehensive Training Dataset...")
        
        # Generate training data
        training_samples = self.generate_comprehensive_training_data(base_samples)
        
        # Convert to ML-ready format
        X, y_quality, y_security = self.create_ml_ready_dataset(training_samples)
        
        # Save everything
        output_dir = self.save_comprehensive_dataset(training_samples, X, y_quality, y_security)
        
        print(f"\n🎉 Comprehensive Training Dataset Complete!")
        print(f"📁 Saved to: {output_dir}")
        
        return output_dir

def main():
    """Main function to create comprehensive training data"""
    creator = ComprehensiveTrainingDataCreator()
    output_dir = creator.create_all(5000)  # 5000 samples
    print(f"🎯 Your comprehensive training data is ready in: {output_dir}")

if __name__ == "__main__":
    main()
