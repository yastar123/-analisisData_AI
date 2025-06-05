import requests
import json
from typing import Dict, Any, Optional
from urllib.parse import urlparse
import os
from dotenv import load_dotenv

load_dotenv()

class APIClient:
    def __init__(self):
        self.timeout = int(os.getenv('REQUEST_TIMEOUT', 30))
        self.max_retries = int(os.getenv('MAX_RETRIES', 3))
        
    def validate_url(self, url: str) -> bool:
        """Validate if the URL is properly formatted"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
            
    def make_request(self, 
                    url: str, 
                    method: str = 'GET',
                    headers: Optional[Dict] = None,
                    params: Optional[Dict] = None,
                    data: Optional[Dict] = None,
                    json_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make an API request with error handling and retries
        
        Args:
            url (str): The API endpoint URL
            method (str): HTTP method (GET, POST, PUT, DELETE)
            headers (dict): Request headers
            params (dict): URL parameters
            data (dict): Form data
            json_data (dict): JSON data
            
        Returns:
            dict: Response with status and data
        """
        if not self.validate_url(url):
            return {
                'success': False,
                'error': 'Invalid URL format'
            }
            
        # Set default headers if none provided
        if headers is None:
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=data,
                json=json_data,
                timeout=self.timeout
            )
            
            # Try to parse JSON response
            try:
                response_data = response.json()
            except:
                response_data = response.text
                
            return {
                'success': True,
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'data': response_data,
                'response_time': response.elapsed.total_seconds()
            }
            
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Request timeout'
            }
            
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'error': 'Connection error'
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'Request error: {str(e)}'
            }
            
    def analyze_api_documentation(self, url: str) -> Dict[str, Any]:
        """
        Analyze API documentation and extract endpoints
        
        Args:
            url (str): URL to API documentation (OpenAPI/Swagger)
            
        Returns:
            dict: Analysis results
        """
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            try:
                doc = response.json()
                
                # Extract basic info
                info = {
                    'title': doc.get('info', {}).get('title', 'Unknown API'),
                    'version': doc.get('info', {}).get('version', 'Unknown'),
                    'description': doc.get('info', {}).get('description', ''),
                    'endpoints': []
                }
                
                # Extract endpoints
                for path, methods in doc.get('paths', {}).items():
                    for method, details in methods.items():
                        endpoint = {
                            'path': path,
                            'method': method.upper(),
                            'summary': details.get('summary', ''),
                            'description': details.get('description', ''),
                            'parameters': details.get('parameters', []),
                            'responses': details.get('responses', {})
                        }
                        info['endpoints'].append(endpoint)
                        
                return {
                    'success': True,
                    'info': info
                }
                
            except json.JSONDecodeError:
                return {
                    'success': False,
                    'error': 'Invalid JSON documentation'
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'Error fetching documentation: {str(e)}'
            }
            
    def test_endpoint(self, 
                     url: str,
                     method: str = 'GET',
                     headers: Optional[Dict] = None,
                     params: Optional[Dict] = None,
                     data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Test an API endpoint and return detailed results
        
        Args:
            url (str): The API endpoint URL
            method (str): HTTP method
            headers (dict): Request headers
            params (dict): URL parameters
            data (dict): Request data
            
        Returns:
            dict: Test results
        """
        result = self.make_request(url, method, headers, params, data)
        
        if result['success']:
            # Add performance metrics
            result['performance'] = {
                'response_time': result.pop('response_time'),
                'status_code': result['status_code']
            }
            
            # Add validation suggestions
            result['suggestions'] = self._generate_suggestions(result)
            
        return result
        
    def _generate_suggestions(self, result: Dict[str, Any]) -> list:
        """Generate suggestions based on API response"""
        suggestions = []
        
        # Check response time
        if result['performance']['response_time'] > 1.0:
            suggestions.append('Response time is slow (>1s). Consider optimizing the endpoint.')
            
        # Check status code
        if result['status_code'] >= 400:
            suggestions.append(f'Endpoint returned error status code {result["status_code"]}.')
            
        # Check content type
        content_type = result['headers'].get('content-type', '')
        if 'application/json' not in content_type:
            suggestions.append('Consider using JSON response format for better compatibility.')
            
        # Check cache headers
        if 'cache-control' not in result['headers']:
            suggestions.append('Consider adding cache-control headers for better performance.')
            
        return suggestions 