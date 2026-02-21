import os


class Config:
    PROXY_TARGET_URL = os.getenv('PROXY_TARGET_URL', 'https://api.anthropic.com')
    PROXY_API_KEY = os.getenv('PROXY_API_KEY', '')
    CODEX_BASE_URL = os.getenv('CODEX_BASE_URL', '')
    CODEX_API_KEY = os.getenv('CODEX_API_KEY', '')
    PROXY_PORT = int(os.getenv('PROXY_PORT', '3029'))
    API_TIMEOUT = int(os.getenv('API_TIMEOUT', '300'))
    ACCESS_API_KEY = os.getenv('ACCESS_API_KEY', '')
