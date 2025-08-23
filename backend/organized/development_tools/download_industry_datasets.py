"""
Download Industry-Standard Datasets for Code Review ML Models

This script downloads and prepares industry-standard datasets including:
- GitHub Security Advisories
- CVE Database
- CodeSearchNet
- BigCloneBench
- VulDeePecker
- Devign
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import zipfile
import tarfile
from tqdm import tqdm

class IndustryDatasetDownloader:
    """Downloads and prepares industry-standard datasets"""
    
    def __init__(self, output_dir: str = "industry_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.datasets = {}
        
    def download_github_security_advisories(self):
        """Download GitHub Security Advisories dataset"""
        print("🔒 Downloading GitHub Security Advisories...")
        
        # GitHub Security Advisories API
        url = "https://api.github.com/advisories"
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "CodeReviewAgent/1.0"
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                advisories = response.json()
                
                # Process advisories
                security_data = []
                for advisory in advisories:
                    if advisory.get('vulnerabilities'):
                        for vuln in advisory['vulnerabilities']:
                            security_data.append({
                                'ghsa_id': advisory.get('ghsa_id'),
                                'cve_id': advisory.get('cve_id'),
                                'severity': advisory.get('severity'),
                                'summary': advisory.get('summary'),
                                'description': advisory.get('description'),
                                'package_name': vuln.get('package', {}).get('name'),
                                'ecosystem': vuln.get('package', {}).get('ecosystem'),
                                'vulnerable_version_range': vuln.get('vulnerable_version_range'),
                                'first_patched_version': vuln.get('first_patched_version'),
                                'published_at': advisory.get('published_at'),
                                'updated_at': advisory.get('updated_at')
                            })
                
                # Save to file
                output_file = self.output_dir / "github_security_advisories.json"
                with open(output_file, 'w') as f:
                    json.dump(security_data, f, indent=2)
                
                print(f"✅ Downloaded {len(security_data)} security advisories")
                self.datasets['github_security'] = {
                    'file': str(output_file),
                    'count': len(security_data),
                    'source': 'GitHub Security API'
                }
                
            else:
                print(f"⚠️ Failed to download GitHub advisories: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error downloading GitHub advisories: {e}")
    
    def download_cve_database(self):
        """Download CVE Database dataset"""
        print("🛡️ Downloading CVE Database...")
        
        # CVE Database download URLs
        cve_urls = {
            '2024': 'https://cve.mitre.org/data/downloads/allitems.csv',
            '2023': 'https://cve.mitre.org/data/downloads/allitems.csv',
            '2022': 'https://cve.mitre.org/data/downloads/allitems.csv'
        }
        
        cve_data = []
        
        for year, url in cve_urls.items():
            try:
                print(f"  Downloading {year} CVEs...")
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    # Parse CSV data
                    lines = response.text.split('\n')
                    for line in lines[1:]:  # Skip header
                        if line.strip():
                            parts = line.split('|')
                            if len(parts) >= 3:
                                cve_data.append({
                                    'cve_id': parts[0].strip(),
                                    'year': year,
                                    'description': parts[2].strip() if len(parts) > 2 else '',
                                    'severity': self._extract_severity(parts[2] if len(parts) > 2 else ''),
                                    'language': self._extract_language(parts[2] if len(parts) > 2 else '')
                                })
                
            except Exception as e:
                print(f"⚠️ Error downloading {year} CVEs: {e}")
        
        if cve_data:
            # Save to file
            output_file = self.output_dir / "cve_database.json"
            with open(output_file, 'w') as f:
                json.dump(cve_data, f, indent=2)
            
            print(f"✅ Downloaded {len(cve_data)} CVE records")
            self.datasets['cve_database'] = {
                'file': str(output_file),
                'count': len(cve_data),
                'source': 'MITRE CVE Database'
            }
    
    def download_codesearchnet(self):
        """Download CodeSearchNet dataset"""
        print("🔍 Downloading CodeSearchNet...")
        
        # CodeSearchNet is available via HuggingFace datasets
        try:
            from datasets import load_dataset
            
            # Load Python subset
            python_dataset = load_dataset("code_search_net", "python", split="train")
            
            # Convert to our format
            code_data = []
            for item in python_dataset:
                code_data.append({
                    'code': item.get('code', ''),
                    'docstring': item.get('docstring', ''),
                    'language': 'python',
                    'repo_name': item.get('repo_name', ''),
                    'path': item.get('path', ''),
                    'url': item.get('url', ''),
                    'license': item.get('license', ''),
                    'size': len(item.get('code', '')),
                    'complexity': self._estimate_complexity(item.get('code', ''))
                })
            
            # Save to file
            output_file = self.output_dir / "codesearchnet_python.json"
            with open(output_file, 'w') as f:
                json.dump(code_data, f, indent=2)
            
            print(f"✅ Downloaded {len(code_data)} CodeSearchNet samples")
            self.datasets['codesearchnet'] = {
                'file': str(output_file),
                'count': len(code_data),
                'source': 'CodeSearchNet (HuggingFace)'
            }
            
        except ImportError:
            print("⚠️ HuggingFace datasets not available, skipping CodeSearchNet")
        except Exception as e:
            print(f"❌ Error downloading CodeSearchNet: {e}")
    
    def download_bigclonebench(self):
        """Download BigCloneBench dataset"""
        print("📊 Downloading BigCloneBench...")
        
        # BigCloneBench download URL
        url = "https://github.com/jeffsvajlenko/BigCloneEval/raw/master/input/input.txt"
        
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # Parse clone data
                clone_data = []
                lines = response.text.split('\n')
                
                for line in lines:
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 4:
                            clone_data.append({
                                'clone_id': parts[0],
                                'file1': parts[1],
                                'file2': parts[2],
                                'clone_type': parts[3],
                                'similarity': float(parts[4]) if len(parts) > 4 else 0.0
                            })
                
                # Save to file
                output_file = self.output_dir / "bigclonebench.json"
                with open(output_file, 'w') as f:
                    json.dump(clone_data, f, indent=2)
                
                print(f"✅ Downloaded {len(clone_data)} BigCloneBench records")
                self.datasets['bigclonebench'] = {
                    'file': str(output_file),
                    'count': len(clone_data),
                    'source': 'BigCloneBench'
                }
                
        except Exception as e:
            print(f"❌ Error downloading BigCloneBench: {e}")
    
    def download_vuldeepecker(self):
        """Download VulDeePecker dataset"""
        print("🕵️ Downloading VulDeePecker...")
        
        # VulDeePecker GitHub repository
        url = "https://github.com/CGCL-codes/VulDeePecker/archive/refs/heads/master.zip"
        
        try:
            response = requests.get(url, timeout=60)
            
            if response.status_code == 200:
                # Save zip file
                zip_file = self.output_dir / "vuldeepecker.zip"
                with open(zip_file, 'wb') as f:
                    f.write(response.content)
                
                # Extract zip file
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(self.output_dir)
                
                # Process extracted data
                vuln_data = self._process_vuldeepecker_data()
                
                # Save processed data
                output_file = self.output_dir / "vuldeepecker_processed.json"
                with open(output_file, 'w') as f:
                    json.dump(vuln_data, f, indent=2)
                
                print(f"✅ Downloaded and processed VulDeePecker dataset")
                self.datasets['vuldeepecker'] = {
                    'file': str(output_file),
                    'count': len(vuln_data),
                    'source': 'VulDeePecker'
                }
                
                # Clean up
                zip_file.unlink()
                
        except Exception as e:
            print(f"❌ Error downloading VulDeePecker: {e}")
    
    def _process_vuldeepecker_data(self) -> List[Dict]:
        """Process extracted VulDeePecker data"""
        vuln_data = []
        
        # Look for vulnerability data in extracted files
        extracted_dir = self.output_dir / "VulDeePecker-master"
        
        if extracted_dir.exists():
            # Process C/C++ vulnerability data
            for file_path in extracted_dir.rglob("*.c"):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    vuln_data.append({
                        'file': str(file_path.relative_to(extracted_dir)),
                        'language': 'c',
                        'content': content[:1000],  # First 1000 chars
                        'size': len(content),
                        'vulnerability_type': 'potential_c_vulnerability',
                        'source': 'VulDeePecker'
                    })
                except:
                    continue
            
            # Process C++ vulnerability data
            for file_path in extracted_dir.rglob("*.cpp"):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    vuln_data.append({
                        'file': str(file_path.relative_to(extracted_dir)),
                        'language': 'cpp',
                        'content': content[:1000],  # First 1000 chars
                        'size': len(content),
                        'vulnerability_type': 'potential_cpp_vulnerability',
                        'source': 'VulDeePecker'
                    })
                except:
                    continue
        
        return vuln_data
    
    def _extract_severity(self, description: str) -> str:
        """Extract severity from CVE description"""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['critical', 'severe', 'high']):
            return 'high'
        elif any(word in description_lower for word in ['medium', 'moderate']):
            return 'medium'
        elif any(word in description_lower for word in ['low', 'minor']):
            return 'low'
        else:
            return 'unknown'
    
    def _extract_language(self, description: str) -> str:
        """Extract programming language from CVE description"""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['python', 'py']):
            return 'python'
        elif any(word in description_lower for word in ['javascript', 'js', 'node']):
            return 'javascript'
        elif any(word in description_lower for word in ['java']):
            return 'java'
        elif any(word in description_lower for word in ['c++', 'cpp', 'c plus plus']):
            return 'cpp'
        elif any(word in description_lower for word in ['c#', 'csharp']):
            return 'csharp'
        else:
            return 'unknown'
    
    def _estimate_complexity(self, code: str) -> int:
        """Estimate code complexity"""
        if not code:
            return 1
        
        complexity = 1
        
        # Count control flow statements
        complexity += code.count('if ')
        complexity += code.count('for ')
        complexity += code.count('while ')
        complexity += code.count('switch ')
        complexity += code.count('catch ')
        complexity += code.count('except ')
        
        return min(complexity, 50)  # Cap at 50
    
    def create_combined_dataset(self):
        """Create a combined dataset from all sources"""
        print("\n🔗 Creating Combined Dataset...")
        
        combined_data = {
            'security_vulnerabilities': [],
            'code_quality': [],
            'metadata': {}
        }
        
        # Combine security data
        if 'github_security' in self.datasets:
            with open(self.datasets['github_security']['file'], 'r') as f:
                github_data = json.load(f)
                combined_data['security_vulnerabilities'].extend(github_data)
        
        if 'cve_database' in self.datasets:
            with open(self.datasets['cve_database']['file'], 'r') as f:
                cve_data = json.load(f)
                combined_data['security_vulnerabilities'].extend(cve_data)
        
        # Combine code quality data
        if 'codesearchnet' in self.datasets:
            with open(self.datasets['codesearchnet']['file'], 'r') as f:
                code_data = json.load(f)
                combined_data['code_quality'].extend(code_data)
        
        # Add metadata
        combined_data['metadata'] = {
            'total_security_vulnerabilities': len(combined_data['security_vulnerabilities']),
            'total_code_samples': len(combined_data['code_quality']),
            'datasets_used': list(self.datasets.keys()),
            'download_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save combined dataset
        combined_file = self.output_dir / "combined_industry_dataset.json"
        with open(combined_file, 'w') as f:
            json.dump(combined_data, f, indent=2)
        
        print(f"✅ Combined dataset created: {combined_file}")
        print(f"📊 Total security vulnerabilities: {combined_data['metadata']['total_security_vulnerabilities']}")
        print(f"📊 Total code samples: {combined_data['metadata']['total_code_samples']}")
        
        return combined_file
    
    def download_all(self):
        """Download all industry datasets"""
        print("🚀 Starting Industry Dataset Download...")
        
        # Download all datasets
        self.download_github_security_advisories()
        self.download_cve_database()
        self.download_codesearchnet()
        self.download_bigclonebench()
        self.download_vuldeepecker()
        
        # Create combined dataset
        combined_file = self.create_combined_dataset()
        
        # Save dataset summary
        summary_file = self.output_dir / "dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.datasets, f, indent=2)
        
        print(f"\n🎉 Download Complete!")
        print(f"📁 All datasets saved to: {self.output_dir}")
        print(f"📊 Combined dataset: {combined_file}")
        
        return self.output_dir

def main():
    """Main function to download industry datasets"""
    downloader = IndustryDatasetDownloader()
    output_dir = downloader.download_all()
    print(f"🎯 Industry datasets ready in: {output_dir}")

if __name__ == "__main__":
    main()
