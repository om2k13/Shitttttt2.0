# 🔍 Code Review Agent

A powerful, automated code review system that analyzes GitHub repositories for code quality, security vulnerabilities, and best practices. This agent supports multiple programming languages and automatically applies safe fixes.

## ✨ Features

### 🚀 **Multi-Language Support**
- **Python**: Ruff, MyPy, Black, Bandit, Radon
- **JavaScript/TypeScript**: ESLint, Prettier, npm audit
- **Universal**: Semgrep (security), detect-secrets
- **Auto-detection** of project type and appropriate tools

### 🔒 **Security Analysis**
- **Vulnerability scanning** with Bandit and Semgrep
- **Secret detection** to prevent credential leaks
- **Dependency vulnerability** analysis (pip-audit, npm audit)
- **Security pattern matching** across all file types

### 🎯 **Code Quality**
- **Static analysis** with industry-standard tools
- **Type checking** for Python and TypeScript
- **Complexity analysis** to identify problematic functions
- **Code formatting** and style enforcement

### 🛠️ **Auto-Fixing**
- **Safe automatic fixes** for code style issues
- **Import organization** and unused import removal
- **Code formatting** to industry standards
- **Non-destructive** - creates fixed versions without modifying originals

## 🏗️ Architecture

```
├── backend/                 # FastAPI backend server
│   ├── app/
│   │   ├── api/            # REST API endpoints
│   │   ├── core/           # Core functionality (LLM, VCS, settings)
│   │   ├── db/             # Database models and connection
│   │   └── review/         # Code review pipeline and tools
│   └── requirements.txt    # Python dependencies
├── frontend/               # React frontend application
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Application pages
│   │   └── lib/            # API client and utilities
│   └── package.json        # Node.js dependencies
└── .workspaces/            # Cloned repositories and analysis results
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### Backend Setup
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

### Start the Application
```bash
# Terminal 1 - Backend
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### Access the Application
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📖 Usage

### 1. **Start Code Review**
- Enter a GitHub repository URL
- Optionally specify a branch
- Click "Start Code Review"

### 2. **Monitor Progress**
- View real-time progress updates
- See current analysis stage
- Track completion percentage

### 3. **Review Results**
- **Summary Dashboard**: Overview of all findings
- **Tool Breakdown**: Issues by analysis tool
- **Severity Analysis**: Critical, High, Medium, Low issues
- **Detailed Findings**: File-by-file issue breakdown

### 4. **Apply Fixes**
- Click "Apply Safe Auto-Fixes"
- Review automatically applied changes
- Compare original vs. fixed versions

## 🛠️ Supported Tools

### **Python Analysis**
- **Ruff**: Fast Python linter with auto-fixing
- **MyPy**: Static type checker
- **Black**: Uncompromising code formatter
- **Bandit**: Security linter
- **Radon**: Code complexity analyzer

### **JavaScript/TypeScript Analysis**
- **ESLint**: JavaScript/TypeScript linting
- **Prettier**: Code formatter
- **npm audit**: Dependency vulnerability scanner

### **Security Tools**
- **Semgrep**: Security pattern matching
- **detect-secrets**: Secret detection
- **pip-audit**: Python dependency vulnerabilities

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the backend directory:

```bash
# LLM Configuration (Optional)
LLM_PROVIDER=openrouter  # or "openai", "none"
LLM_MODEL_ID=qwen/Qwen2.5-7B-Instruct
OPENROUTER_API_KEY=your_api_key_here

# Database
DATABASE_URL=sqlite+aiosqlite:///./agent.db

# GitHub (Optional)
GITHUB_TOKEN=your_github_token_here
```

### LLM Integration
The agent can optionally use LLM models to:
- Provide detailed explanations for findings
- Suggest specific remediation steps
- Enhance issue descriptions with context

## 📊 Sample Output

### Dashboard View
```
🔍 Code Review Agent
Repository: https://github.com/username/project
Status: ✅ Completed
Issues Found: 74

📊 Summary
├── Total Issues: 74
├── Critical: 0
├── High: 2
├── Medium: 58
└── Low: 14

🛠️ Tools Used
├── Ruff (Code Style): 16 issues
├── MyPy (Type Checking): 57 issues
└── Semgrep (Security): 1 issue
```

### Detailed Findings
Each finding includes:
- **File path** and line number
- **Issue description** and severity
- **Tool** that found the issue
- **Rule ID** for reference
- **Auto-fixable** status

## 🎯 Use Cases

### **Development Teams**
- **Code quality enforcement** in CI/CD pipelines
- **Security vulnerability** prevention
- **Consistent coding standards** across projects
- **Automated code review** for pull requests

### **Open Source Projects**
- **Quality assurance** for contributions
- **Security scanning** for vulnerabilities
- **Documentation** of code quality metrics

### **Learning & Education**
- **Code review practice** for developers
- **Best practices** identification
- **Security awareness** training

## 🔒 Security Features

- **Safe analysis**: No code execution, only static analysis
- **Isolated workspaces**: Each repository analyzed in separate environment
- **No data persistence**: Analysis results stored locally only
- **Secure tool execution**: All tools run in controlled environment

## 🚀 Performance

- **Fast analysis**: Parallel tool execution
- **Efficient cloning**: Git-based repository access
- **Smart caching**: Tool results cached for performance
- **Scalable architecture**: Can handle multiple concurrent reviews

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ruff**: Fast Python linter
- **MyPy**: Python type checker
- **Semgrep**: Security pattern matching
- **FastAPI**: Modern Python web framework
- **React**: Frontend framework

## 📞 Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review the API documentation at `/docs`

---

**Built with ❤️ for the developer community**
