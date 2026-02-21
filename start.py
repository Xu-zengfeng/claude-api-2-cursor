import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)

from config import Config
from app import create_app

if __name__ == '__main__':
    app = create_app()
    print(f'Proxy service starting on 0.0.0.0:{Config.PROXY_PORT}')
    print(f'Target: {Config.PROXY_TARGET_URL}')
    # 启动时检查环境变量是否读到了（便于排查上游 503 no available accounts）
    key_ok = bool(Config.PROXY_API_KEY and Config.PROXY_API_KEY.strip())
    print(f'PROXY_API_KEY: {"✓ set" if key_ok else "✗ missing or empty"}')
    if not key_ok:
        print('WARNING: PROXY_API_KEY is empty. Requests to upstream may fail.')

    from waitress import serve
    serve(
        app,
        host='0.0.0.0',
        port=Config.PROXY_PORT,
        channel_timeout=Config.API_TIMEOUT,
        send_bytes=1,
    )
