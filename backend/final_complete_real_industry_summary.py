import json
from pathlib import Path

def create_final_complete_summary():
    """Create final comprehensive summary of ALL REAL industry data"""
    
    # Initialize totals
    total_real_samples = 0
    total_security_samples = 0
    total_quality_rules = 0
    
    # Load VulDeePecker data
    try:
        with open('industry_datasets/vuldeepecker_processed.json', 'r') as f:
            vuldeepecker_data = json.load(f)
            vuldeepecker_count = len(vuldeepecker_data)
            total_security_samples += vuldeepecker_count
            total_real_samples += vuldeepecker_count
            print(f"✅ VulDeePecker: {vuldeepecker_count:,} security samples")
    except Exception as e:
        print(f"❌ Error loading VulDeePecker: {e}")
        vuldeepecker_count = 0
    
    # Load NIST NVD CVE data
    try:
        with open('real_industry_data/real_nvd_cve_database.json', 'r') as f:
            nvd_data = json.load(f)
            nvd_count = len(nvd_data)
            total_security_samples += nvd_count
            total_real_samples += nvd_count
            print(f"✅ NIST NVD CVE: {nvd_count:,} security samples")
    except Exception as e:
        print(f"❌ Error loading NVD CVE: {e}")
        nvd_count = 0
    
    # Load MITRE CWE data
    try:
        with open('real_industry_data/real_mitre_cwe_database.json', 'r') as f:
            cwe_data = json.load(f)
            cwe_count = len(cwe_data)
            total_security_samples += cwe_count
            total_real_samples += cwe_count
            print(f"✅ MITRE CWE: {cwe_count:,} security samples")
    except Exception as e:
        print(f"❌ Error loading MITRE CWE: {e}")
        cwe_count = 0
    
    # Load Debian CVE data (NEW!)
    try:
        with open('real_debian_cve_database.json', 'r') as f:
            debian_data = json.load(f)
            debian_count = len(debian_data)
            total_security_samples += debian_count
            total_real_samples += debian_count
            print(f"✅ Debian Security Tracker: {debian_count:,} security samples")
    except Exception as e:
        print(f"❌ Error loading Debian CVE: {e}")
        debian_count = 0
    
    # Load ESLint rules
    try:
        with open('real_industry_data/real_eslint_rules.json', 'r') as f:
            eslint_data = json.load(f)
            eslint_count = len(eslint_data)
            total_quality_rules += eslint_count
            total_real_samples += eslint_count
            print(f"✅ ESLint Rules: {eslint_count} quality rules")
    except Exception as e:
        print(f"❌ Error loading ESLint: {e}")
        eslint_count = 0
    
    # Load Bandit rules
    try:
        with open('real_industry_data/real_bandit_rules.json', 'r') as f:
            bandit_data = json.load(f)
            bandit_count = len(bandit_data)
            total_quality_rules += bandit_count
            total_real_samples += bandit_count
            print(f"✅ Bandit Rules: {bandit_count} quality rules")
    except Exception as e:
        print(f"❌ Error loading Bandit: {e}")
        bandit_count = 0
    
    # Load CodeQL rules
    try:
        with open('real_industry_data/real_codeql_rules_fixed.json', 'r') as f:
            codeql_data = json.load(f)
            codeql_count = len(codeql_data)
            total_quality_rules += codeql_count
            total_real_samples += codeql_count
            print(f"✅ CodeQL Rules: {codeql_count} quality rules")
    except Exception as e:
        print(f"❌ Error loading CodeQL: {e}")
        codeql_count = 0
    
    # Load OWASP Top 10
    try:
        with open('real_industry_data/real_owasp_top10.json', 'r') as f:
            owasp_data = json.load(f)
            owasp_count = len(owasp_data)
            total_quality_rules += owasp_count
            total_real_samples += owasp_count
            print(f"✅ OWASP Top 10: {owasp_count} quality rules")
    except Exception as e:
        print(f"❌ Error loading OWASP: {e}")
        owasp_count = 0
    
    # Create comprehensive summary
    summary = {
        "total_real_samples": total_real_samples,
        "total_security_samples": total_security_samples,
        "total_quality_rules": total_quality_rules,
        "datasets": {
            "vuldeepecker": {
                "count": vuldeepecker_count,
                "type": "security_samples",
                "source": "VulDeePecker Dataset"
            },
            "nist_nvd_cve": {
                "count": nvd_count,
                "type": "security_samples", 
                "source": "NIST National Vulnerability Database"
            },
            "mitre_cwe": {
                "count": cwe_count,
                "type": "security_samples",
                "source": "MITRE Common Weakness Enumeration"
            },
            "debian_cve": {
                "count": debian_count,
                "type": "security_samples",
                "source": "Debian Security Tracker"
            },
            "eslint_rules": {
                "count": eslint_count,
                "type": "quality_rules",
                "source": "ESLint Security Rules"
            },
            "bandit_rules": {
                "count": bandit_count,
                "type": "quality_rules",
                "source": "Bandit Security Rules"
            },
            "codeql_rules": {
                "count": codeql_count,
                "type": "quality_rules",
                "source": "GitHub CodeQL Security Queries"
            },
            "owasp_top10": {
                "count": owasp_count,
                "type": "quality_rules",
                "source": "OWASP Top 10 2021"
            }
        },
        "industry_standards": {
            "security_samples_target": 50000,
            "security_samples_achieved": total_security_samples,
            "security_samples_status": "EXCEEDED" if total_security_samples >= 50000 else "INCOMPLETE",
            "quality_rules_target": 1000,
            "quality_rules_achieved": total_quality_rules,
            "quality_rules_status": "EXCEEDED" if total_quality_rules >= 1000 else "INCOMPLETE"
        },
        "official_sources": [
            "VulDeePecker Academic Dataset",
            "NIST National Vulnerability Database",
            "MITRE Common Weakness Enumeration", 
            "Debian Security Tracker",
            "ESLint Security Rules",
            "Bandit Security Rules",
            "GitHub CodeQL Security Queries",
            "OWASP Top 10 2021"
        ]
    }
    
    # Save summary
    with open('final_complete_real_industry_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print achievement summary
    print("\n" + "="*80)
    print("🏆 FINAL REAL INDUSTRY DATA ACHIEVEMENT SUMMARY")
    print("="*80)
    print(f"📊 Total REAL Samples: {total_real_samples:,}")
    print(f"🛡️  Total REAL Security Samples: {total_security_samples:,}")
    print(f"🔧 Total REAL Quality Rules: {total_quality_rules}")
    print(f"📚 Official Sources: {len(summary['official_sources'])}")
    print()
    
    # Security samples achievement
    if total_security_samples >= 50000:
        print(f"✅ SECURITY SAMPLES: EXCEEDED Industry Standard (50,000+)")
        print(f"   Current: {total_security_samples:,} samples")
        print(f"   Target: 50,000+ samples")
        print(f"   Achievement: {total_security_samples - 50000:,} samples ABOVE standard")
    else:
        print(f"❌ SECURITY SAMPLES: INCOMPLETE")
        print(f"   Current: {total_security_samples:,} samples")
        print(f"   Target: 50,000+ samples")
        print(f"   Missing: {50000 - total_security_samples:,} samples")
    
    print()
    
    # Quality rules achievement
    if total_quality_rules >= 1000:
        print(f"✅ QUALITY RULES: EXCEEDED Industry Standard (1,000+)")
        print(f"   Current: {total_quality_rules} rules")
        print(f"   Target: 1,000+ rules")
        print(f"   Achievement: {total_quality_rules - 1000} rules ABOVE standard")
    else:
        print(f"❌ QUALITY RULES: INCOMPLETE")
        print(f"   Current: {total_quality_rules} rules")
        print(f"   Target: 1,000+ rules")
        print(f"   Missing: {1000 - total_quality_rules} rules")
    
    print()
    print("🎯 MISSION STATUS: REAL INDUSTRY DATA COLLECTION")
    print("="*80)
    
    if total_security_samples >= 50000 and total_quality_rules >= 1000:
        print("🏆 MISSION ACCOMPLISHED - FULL INDUSTRY STANDARDS ACHIEVED!")
        print("🚀 Ready to build production ML models with real industry data!")
    elif total_security_samples >= 50000:
        print("🟡 PARTIAL SUCCESS - Security samples achieved, quality rules pending")
        print("📈 Need more quality rules to reach full industry standards")
    else:
        print("🔴 INCOMPLETE - Still need more data to reach industry standards")
    
    print("="*80)
    
    return summary

if __name__ == "__main__":
    create_final_complete_summary()
