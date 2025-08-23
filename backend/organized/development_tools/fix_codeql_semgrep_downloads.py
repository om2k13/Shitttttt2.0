"""
Fix CodeQL and Semgrep Downloads

Uses the correct repository structure I discovered:
- CodeQL: javascript/ql/src/Security/CWE-XXX directories
- Semgrep: returntocorp/semgrep-rules structure
"""

import json
import requests
import time
from pathlib import Path

class FixCodeQLSemgrepDownloads:
    def __init__(self, base_dir: str = "real_industry_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def download_codeql_rules_fixed(self):
        """Download CodeQL rules using the correct path structure"""
        print("🔍 Downloading CodeQL Rules (Fixed Path Structure)...")
        
        # CodeQL repository (official GitHub repo)
        codeql_repo_url = "https://api.github.com/repos/github/codeql/contents"
        
        try:
            headers = {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "CodeReviewAgent/1.0"
            }
            
            # Languages to process
            languages = ['javascript', 'python', 'java', 'go', 'cpp']
            codeql_rules = []
            
            for language in languages:
                print(f"  Processing {language} language...")
                
                # Get language directory contents
                lang_url = f"{codeql_repo_url}/{language}/ql/src"
                lang_response = requests.get(lang_url, headers=headers, timeout=30)
                
                if lang_response.status_code == 200:
                    lang_dirs = lang_response.json()
                    
                    # Look for Security directory
                    for dir_info in lang_dirs:
                        if dir_info.get('name') == 'Security' and dir_info.get('type') == 'dir':
                            print(f"    Found Security directory in {language}")
                            
                            # Get Security directory contents
                            security_url = dir_info['url']
                            security_response = requests.get(security_url, headers=headers, timeout=30)
                            
                            if security_response.status_code == 200:
                                security_dirs = security_response.json()
                                
                                for security_dir in security_dirs:
                                    if security_dir.get('type') == 'dir':
                                        # Get contents of this security subdirectory
                                        subdir_url = security_dir['url']
                                        subdir_response = requests.get(subdir_url, headers=headers, timeout=30)
                                        
                                        if subdir_response.status_code == 200:
                                            subdir_files = subdir_response.json()
                                            
                                            for file_info in subdir_files:
                                                if file_info.get('name', '').endswith('.ql'):
                                                    # Download the actual query file
                                                    if file_info.get('download_url'):
                                                        try:
                                                            file_response = requests.get(file_info['download_url'], timeout=30)
                                                            
                                                            if file_response.status_code == 200:
                                                                codeql_rules.append({
                                                                    'name': file_info['name'],
                                                                    'path': file_info['path'],
                                                                    'language': language,
                                                                    'size': file_info['size'],
                                                                    'content': file_response.text[:2000],  # First 2000 chars
                                                                    'download_url': file_info['download_url'],
                                                                    'source': 'GitHub CodeQL (Official Repository)',
                                                                    'security_category': security_dir['name']
                                                                })
                                                                
                                                                print(f"      Downloaded {file_info['name']}")
                                                            
                                                            time.sleep(0.1)  # Small delay
                                                            
                                                        except Exception as e:
                                                            print(f"      Error downloading {file_info['name']}: {e}")
                                                            continue
                                            
                                            time.sleep(0.5)  # Delay between subdirectories
                                    
                            break  # Found Security directory, move to next language
                        
                        time.sleep(0.2)  # Small delay
                
                time.sleep(1)  # Delay between languages
            
            # Save CodeQL rules
            if codeql_rules:
                output_file = self.base_dir / "real_codeql_rules_fixed.json"
                with open(output_file, 'w') as f:
                    json.dump(codeql_rules, f, indent=2)
                
                print(f"✅ Downloaded {len(codeql_rules)} REAL CodeQL rules (Fixed)")
                return codeql_rules
            else:
                print("❌ No CodeQL rules found")
                
        except Exception as e:
            print(f"❌ Error downloading CodeQL rules: {e}")
        
        return []
    
    def download_semgrep_rules_fixed(self):
        """Download Semgrep rules using the correct structure"""
        print("📋 Downloading Semgrep Rules (Fixed Structure)...")
        
        # Semgrep rules repository (official)
        semgrep_api_url = "https://api.github.com/repos/returntocorp/semgrep-rules/contents"
        
        try:
            headers = {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "CodeReviewAgent/1.0"
            }
            
            response = requests.get(semgrep_api_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                dirs = response.json()
                
                semgrep_rules = []
                language_dirs = ['python', 'javascript', 'java', 'go', 'php', 'ruby']
                
                for dir_info in dirs:
                    if dir_info.get('name') in language_dirs and dir_info.get('type') == 'dir':
                        print(f"  Processing {dir_info['name']} language...")
                        
                        # Get rules from this language directory
                        lang_url = dir_info['url']
                        lang_response = requests.get(lang_url, headers=headers, timeout=30)
                        
                        if lang_response.status_code == 200:
                            lang_files = lang_response.json()
                            
                            for file_info in lang_files:
                                if file_info.get('name', '').endswith('.yml') or file_info.get('name', '').endswith('.yaml'):
                                    # Download the rule file
                                    if file_info.get('download_url'):
                                        try:
                                            file_response = requests.get(file_info['download_url'], timeout=30)
                                            
                                            if file_response.status_code == 200:
                                                semgrep_rules.append({
                                                    'name': file_info['name'],
                                                    'path': file_info['path'],
                                                    'language': dir_info['name'],
                                                    'size': file_info['size'],
                                                    'content': file_response.text[:2000],  # First 2000 chars
                                                    'download_url': file_info['download_url'],
                                                    'source': 'Semgrep Rules (Official Repository)'
                                                })
                                                
                                                print(f"    Downloaded {file_info['name']}")
                                            
                                            time.sleep(0.1)  # Small delay
                                            
                                        except Exception as e:
                                            print(f"    Error downloading {file_info['name']}: {e}")
                                            continue
                        
                        time.sleep(0.5)  # Delay between directories
                
                # Save Semgrep rules
                if semgrep_rules:
                    output_file = self.base_dir / "real_semgrep_rules_fixed.json"
                    with open(output_file, 'w') as f:
                        json.dump(semgrep_rules, f, indent=2)
                    
                    print(f"✅ Downloaded {len(semgrep_rules)} REAL Semgrep rules (Fixed)")
                    return semgrep_rules
                else:
                    print("❌ No Semgrep rules found")
                    
        except Exception as e:
            print(f"❌ Error downloading Semgrep rules: {e}")
        
        return []
    
    def download_all_fixed(self):
        """Download all fixed datasets"""
        print("🚀 Starting Fixed Downloads...")
        print("=" * 50)
        
        # Download fixed datasets
        codeql_rules = self.download_codeql_rules_fixed()
        semgrep_rules = self.download_semgrep_rules_fixed()
        
        print(f"\n🎉 Fixed Downloads Complete!")
        print(f"📊 CodeQL Rules: {len(codeql_rules)}")
        print(f"📊 Semgrep Rules: {len(semgrep_rules)}")
        
        return {
            'codeql': codeql_rules,
            'semgrep': semgrep_rules
        }

def main():
    """Main function"""
    downloader = FixCodeQLSemgrepDownloads()
    results = downloader.download_all_fixed()
    print(f"🎯 Fixed downloads ready!")

if __name__ == "__main__":
    main()
