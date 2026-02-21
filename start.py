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
    logger = logging.getLogger('startup')
    app = create_app()
    logger.info('Proxy starting on 0.0.0.0:%s', Config.PROXY_PORT)
    logger.info('Target: %s', Config.PROXY_TARGET_URL)
    key_ok = bool(Config.PROXY_API_KEY and Config.PROXY_API_KEY.strip())
    logger.info('PROXY_API_KEY: %s', 'set' if key_ok else 'missing or empty')
    if not key_ok:
        logger.warning('PROXY_API_KEY is empty; upstream requests may fail.')

    from waitress import serve
    serve(
        app,
        host='0.0.0.0',
        port=Config.PROXY_PORT,
        channel_timeout=Config.API_TIMEOUT,
        send_bytes=1,
    )
