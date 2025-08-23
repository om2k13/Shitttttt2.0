"""
Collect Real Training Data from Code Repository

This script analyzes actual code files to collect real training data for ML models.
"""

import json
import ast
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

class CodeAnalyzer:
    """Analyzes actual code files to extract real metrics and patterns"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.python_files = []
        self.findings = []
        
    def collect_files(self):
        """Collect Python files in the repository"""
        print(f"🔍 Scanning repository: {self.repo_path}")
        
        # Collect Python files (exclude venv and other system directories)
        for py_file in self.repo_path.rglob("*.py"):
            # Skip virtual environment and system files
            skip_file = False
            for part in py_file.parts:
                if part in ['venv', '.venv', '__pycache__', '.git', 'node_modules']:
                    skip_file = True
                    break
                if part.startswith('.'):
                    skip_file = True
                    break
            
            if not skip_file:
                self.python_files.append(py_file)
        
        print(f"📁 Found {len(self.python_files)} Python files")
    
    def analyze_python_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a Python file for metrics and patterns"""
        try:
            # Try to read file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                print(f"⚠️ Could not read {file_path}")
                return None
            
            # Parse AST
            tree = ast.parse(content)
            
            # Extract metrics
            metrics = {
                'lines': len(content.splitlines()),
                'complexity': self._calculate_cyclomatic_complexity(tree),
                'nesting': self._calculate_max_nesting(tree),
                'imports': len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]),
                'functions': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                'classes': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
            }
            
            # Identify patterns
            patterns = self._identify_python_patterns(content, tree)
            
            # Calculate quality score
            quality_score = self._calculate_python_quality(metrics, patterns)
            
            return {
                'file': str(file_path.relative_to(self.repo_path)),
                'language': 'python',
                'metrics': metrics,
                'patterns': patterns,
                'quality_score': quality_score,
                'issues': self._identify_issues(metrics, patterns)
            }
            
        except Exception as e:
            print(f"⚠️ Error analyzing {file_path}: {e}")
            return None
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity from AST"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _calculate_max_nesting(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0
        current_depth = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.Try, ast.With)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif isinstance(node, ast.FunctionDef):
                current_depth = 0
        
        return max_depth
    
    def _identify_python_patterns(self, content: str, tree: ast.AST) -> List[str]:
        """Identify Python-specific patterns"""
        patterns = []
        
        # Security patterns
        if 'os.system(' in content or 'subprocess.call(' in content:
            patterns.append('command_injection')
        if 'eval(' in content:
            patterns.append('code_injection')
        if 'pickle.loads(' in content:
            patterns.append('deserialization_vulnerability')
        
        # Quality patterns
        if len(content.splitlines()) > 100:
            patterns.append('long_file')
        
        return patterns
    
    def _calculate_python_quality(self, metrics: Dict, patterns: List[str]) -> float:
        """Calculate quality score for Python code"""
        score = 0.8  # Base score
        
        # Penalize based on metrics
        if metrics['lines'] > 200:
            score -= 0.2
        elif metrics['lines'] > 100:
            score -= 0.1
        
        if metrics['complexity'] > 20:
            score -= 0.3
        elif metrics['complexity'] > 10:
            score -= 0.15
        
        if metrics['nesting'] > 5:
            score -= 0.2
        elif metrics['nesting'] > 3:
            score -= 0.1
        
        # Penalize based on patterns
        for pattern in patterns:
            if pattern in ['command_injection', 'code_injection']:
                score -= 0.4
            elif pattern in ['long_file']:
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _identify_issues(self, metrics: Dict, patterns: List[str]) -> List[str]:
        """Identify code quality and security issues"""
        issues = []
        
        # Code quality issues
        if metrics['lines'] > 100:
            issues.append('file_too_long')
        if metrics['complexity'] > 15:
            issues.append('high_complexity')
        if metrics['nesting'] > 4:
            issues.append('deep_nesting')
        
        # Security issues
        for pattern in patterns:
            if 'injection' in pattern or 'vulnerability' in pattern:
                issues.append('security_risk')
        
        return issues
    
    def analyze_all_files(self):
        """Analyze all collected files"""
        print("🔍 Analyzing Python files...")
        python_results = []
        for py_file in self.python_files:
            result = self.analyze_python_file(py_file)
            if result:
                python_results.append(result)
        
        self.findings = python_results
        print(f"✅ Analyzed {len(self.findings)} files")
    
    def generate_ml_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate ML dataset from real code analysis"""
        if not self.findings:
            raise ValueError("No findings available. Run analyze_all_files() first.")
        
        X = []
        y_quality = []
        y_security = []
        
        for finding in self.findings:
            metrics = finding['metrics']
            
            # Create feature vector
            features = [
                metrics['lines'],
                metrics['complexity'],
                metrics['nesting'],
                metrics['imports'],
                metrics['functions'],
                metrics['classes']
            ]
            
            X.append(features)
            
            # Quality labels (0=high, 1=medium, 2=low)
            if finding['quality_score'] >= 0.8:
                y_quality.append(0)
            elif finding['quality_score'] >= 0.6:
                y_quality.append(1)
            else:
                y_quality.append(2)
            
            # Security labels (0=safe, 1=risky)
            has_security_issue = any('security' in issue for issue in finding['issues'])
            y_security.append(1 if has_security_issue else 0)
        
        return np.array(X), np.array(y_quality), np.array(y_security)
    
    def save_results(self, output_dir: str = "real_code_analysis"):
        """Save analysis results and ML dataset"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed analysis
        with open(output_path / "code_analysis.json", "w") as f:
            json.dump(self.findings, f, indent=2)
        
        # Generate and save ML dataset
        X, y_quality, y_security = self.generate_ml_dataset()
        
        np.save(output_path / "features.npy", X)
        np.save(output_path / "quality_labels.npy", y_quality)
        np.save(output_path / "security_labels.npy", y_security)
        
        # Save metadata
        metadata = {
            "total_files": len(self.findings),
            "python_files": len([f for f in self.findings if f['language'] == 'python']),
            "dataset_size": len(X),
            "features": X.shape[1],
            "quality_distribution": {
                "high": int(np.sum(y_quality == 0)),
                "medium": int(np.sum(y_quality == 1)),
                "low": int(np.sum(y_quality == 2))
            },
            "security_distribution": {
                "safe": int(np.sum(y_security == 0)),
                "risky": int(np.sum(y_security == 1))
            }
        }
        
        with open(output_path / "dataset_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Results saved to {output_path}")
        print(f"📊 Dataset: {len(X)} samples with {X.shape[1]} features")
        print(f"📊 Quality distribution: {metadata['quality_distribution']}")
        print(f"📊 Security distribution: {metadata['security_distribution']}")
        
        return output_path

def main():
    """Main function to collect real training data"""
    # Analyze current repository
    analyzer = CodeAnalyzer(".")
    analyzer.collect_files()
    analyzer.analyze_all_files()
    
    # Save results
    output_dir = analyzer.save_results()
    print(f"🎯 Real training data ready in: {output_dir}")

if __name__ == "__main__":
    main()
