import requests
import os
import json
from typing import Dict, List
from dotenv import load_dotenv
import base64
import mimetypes
import PyPDF2
import docx
from PIL import Image
import io
import pandas as pd
import csv

load_dotenv()

class OpenRouterClient:
    def __init__(self):
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv('HTTP_REFERER', 'https://example.com'),
            "X-Title": os.getenv('X_TITLE', 'My AI Agent')
        }
        
        # Model parameters
        self.max_tokens = int(os.getenv('MAX_TOKENS', 1000))
        self.temperature = float(os.getenv('TEMPERATURE', 0.7))
        self.top_p = float(os.getenv('TOP_P', 0.9))
        self.timeout = int(os.getenv('REQUEST_TIMEOUT', 30))

    def read_file_content(self, file_path):
        """Read content from various file types"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext in ['.jpg', '.jpeg', '.png', '.gif']:
                # Handle images
                with Image.open(file_path) as img:
                    # Convert image to base64
                    buffered = io.BytesIO()
                    img.save(buffered, format=img.format)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    return f"[Image content: {img_str}]"
                    
            elif file_ext == '.pdf':
                # Handle PDF files
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
                    
            elif file_ext in ['.doc', '.docx']:
                # Handle Word documents
                doc = docx.Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
                
            elif file_ext == '.txt':
                # Handle text files
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
                    
            elif file_ext in ['.xlsx', '.xls']:
                # Handle Excel files
                df = pd.read_excel(file_path)
                return df.to_string()
                
            elif file_ext == '.csv':
                # Handle CSV files
                df = pd.read_csv(file_path)
                return df.to_string()
                
            elif file_ext in ['.mp4', '.mov']:
                # For video files, we'll just return metadata
                return f"[Video file: {os.path.basename(file_path)}]"
                
            else:
                return f"[Unsupported file type: {file_ext}]"
                
        except Exception as e:
            return f"[Error reading file: {str(e)}]"

    def generate_response(self, prompt, model_id, file_path=None):
        """
        Generate a response from the specified model
        
        Args:
            prompt (str): The user's input prompt
            model_id (str): The model ID to use
            file_path (str, optional): Path to the file to analyze
            
        Returns:
            str: The model's response
        """
        try:
            # If file is provided, read its content
            if file_path:
                file_content = self.read_file_content(file_path)
                prompt = f"File content:\n{file_content}\n\nUser question: {prompt}"

            print(f"Using model_id: {model_id}")  # Tambahkan logging model_id

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "stream": True  # Enable streaming response
                },
                timeout=self.timeout,
                stream=True  # Enable streaming
            )
            
            response.raise_for_status()
            
            # Process streaming response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            if 'choices' in data and len(data['choices']) > 0:
                                content = data['choices'][0].get('delta', {}).get('content', '')
                                if content:
                                    full_response += content
                                    yield content
                        except json.JSONDecodeError:
                            continue
            
            return full_response
            
        except requests.exceptions.RequestException as e:
            try:
                error_msg = e.response.json().get('error', {}).get('message', str(e))
            except Exception:
                error_msg = e.response.text if e.response is not None else str(e)
            raise Exception(f"API Error: {error_msg}")
        
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

class LLMClient:
    def __init__(self):
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Available models
        self.models = {
            "deepseek": "deepseek/deepseek-chat-v3-0324:free",
            "gemini": "google/gemini-2.0-flash-exp:free", 
            "mistral": "mistralai/devstral-small:free"
        }
        
        # Default model
        self.default_model = os.getenv('DEFAULT_MODEL', 'deepseek')
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    
    def get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv('HTTP_REFERER', 'https://example.com'),
            "X-Title": os.getenv('X_TITLE', 'My AI Agent')
        }
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available models"""
        return [
            {"key": "deepseek", "name": "DeepSeek Chat V3", "model": self.models["deepseek"]},
            {"key": "gemini", "name": "Google Gemini 2 Flash", "model": self.models["gemini"]},
            {"key": "mistral", "name": "Mistral Devstral Small", "model": self.models["mistral"]}
        ]
    
    def chat(self, message: str, model_key: str = None) -> Dict:
        """
        Send chat message to OpenRouter API
        
        Args:
            message (str): User message
            model_key (str): Model key (deepseek, gemini, mistral)
        
        Returns:
            Dict: Response with success status and data
        """
        try:
            # Use default model if none specified
            if not model_key or model_key not in self.models:
                model_key = self.default_model
            
            model_name = self.models[model_key]
            
            # Prepare request payload
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                "max_tokens": int(os.getenv('MAX_TOKENS', 1000)),
                "temperature": float(os.getenv('TEMPERATURE', 0.7)),
                "top_p": float(os.getenv('TOP_P', 0.9))
            }
            
            # Make request
            response = requests.post(
                self.base_url,
                headers=self.get_headers(),
                json=payload,
                timeout=int(os.getenv('REQUEST_TIMEOUT', 30))
            )
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                
                if 'choices' in data and len(data['choices']) > 0:
                    ai_response = data['choices'][0]['message']['content']
                    
                    return {
                        'success': True,
                        'response': ai_response,
                        'model_used': model_name,
                        'usage': data.get('usage', {}),
                        'model_key': model_key
                    }
                else:
                    return {
                        'success': False,
                        'error': 'No response from AI model'
                    }
            
            elif response.status_code == 401:
                return {
                    'success': False,
                    'error': 'API key tidak valid atau expired'
                }
            
            elif response.status_code == 429:
                return {
                    'success': False,
                    'error': 'Rate limit exceeded. Coba lagi dalam beberapa saat.'
                }
            
            elif response.status_code == 400:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get('error', {}).get('message', 'Bad request')
                return {
                    'success': False,
                    'error': f'Bad request: {error_msg}'
                }
            
            else:
                return {
                    'success': False,
                    'error': f'HTTP {response.status_code}: {response.text}'
                }
        
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Request timeout. Server mungkin sedang lambat.'
            }
        
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'error': 'Koneksi gagal. Periksa koneksi internet Anda.'
            }
        
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'Request error: {str(e)}'
            }
        
        except json.JSONDecodeError:
            return {
                'success': False,
                'error': 'Invalid JSON response from API'
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}'
            }
    
    def test_connection(self) -> Dict:
        """Test API connection"""
        return self.chat("Hello, can you respond with 'Connection successful'?")

# Example usage and testing
if __name__ == "__main__":
    # Load environment for testing
    from dotenv import load_dotenv
    load_dotenv()
    
    client = LLMClient()
    
    print("🧪 Testing LLM Client...")
    print("Available models:", [m['name'] for m in client.get_available_models()])
    
    # Test connection
    result = client.test_connection()
    if result['success']:
        print("✅ Connection test successful!")
        print("Response:", result['response'])
    else:
        print("❌ Connection test failed:", result['error'])