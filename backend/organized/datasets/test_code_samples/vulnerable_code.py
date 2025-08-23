
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
