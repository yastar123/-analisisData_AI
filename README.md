# AI Chat Agent

A simple web-based AI chat interface that uses OpenRouter API to communicate with various LLM models.

## Features

- Chat interface with multiple AI models:
  - DeepSeek Chat V3
  - Google Gemini 2 Flash
  - Mistral Devstral Small
- Real-time chat updates
- Model selection dropdown
- Error handling and input validation
- Responsive design using Tailwind CSS

## Local Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-agent
```

2. Create a virtual environment and activate it:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your configuration:
```env
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_api_key_here

# Default Model (deepseek, gemini, mistral)
DEFAULT_MODEL=deepseek

# API Request Headers
HTTP_REFERER=https://example.com
X_TITLE=My AI Agent

# Model Parameters
MAX_TOKENS=1000
TEMPERATURE=0.7
TOP_P=0.9
REQUEST_TIMEOUT=30

# Server Configuration
PORT=5000
DEBUG=False

# Security
FLASK_SECRET_KEY=your-secret-key-here-change-this-in-production
```

5. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Deployment to VPS

1. Set up your VPS with Python 3.8+ and install required packages:
```bash
sudo apt update
sudo apt install python3-venv nginx
```

2. Clone the repository to your VPS:
```bash
git clone <repository-url>
cd ai-agent
```

3. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Create and configure the `.env` file as shown in the local setup section.

6. Set up Gunicorn as a service:
```bash
sudo nano /etc/systemd/system/ai-agent.service
```

Add the following content:
```ini
[Unit]
Description=AI Chat Agent
After=network.target

[Service]
User=your_username
WorkingDirectory=/path/to/ai-agent
Environment="PATH=/path/to/ai-agent/venv/bin"
ExecStart=/path/to/ai-agent/venv/bin/gunicorn --workers 3 --bind 0.0.0.0:5000 app:app

[Install]
WantedBy=multi-user.target
```

7. Start and enable the service:
```bash
sudo systemctl start ai-agent
sudo systemctl enable ai-agent
```

8. Configure Nginx as a reverse proxy:
```bash
sudo nano /etc/nginx/sites-available/ai-agent
```

Add the following configuration:
```nginx
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

9. Enable the site and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/ai-agent /etc/nginx/sites-enabled
sudo nginx -t
sudo systemctl restart nginx
```

## Security Considerations

1. Always use HTTPS in production
2. Change the `FLASK_SECRET_KEY` in production
3. Set appropriate file permissions
4. Keep your API keys secure
5. Regularly update dependencies

## Customization

To change the default model, modify the `DEFAULT_MODEL` value in the `.env` file. Available options are:
- `deepseek`
- `gemini`
- `mistral`

## Troubleshooting

1. If the application fails to start, check:
   - API key is correctly set in `.env`
   - All dependencies are installed
   - Port 5000 is not in use

2. For deployment issues:
   - Check Gunicorn logs: `sudo journalctl -u ai-agent`
   - Check Nginx logs: `sudo tail -f /var/log/nginx/error.log`
   - Ensure firewall allows traffic on port 80/443 