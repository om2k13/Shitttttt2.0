
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
