# run_waitress.py
import logging
from waitress import serve
from app import app

# Configure waitress logging to only show errors
logger = logging.getLogger('waitress')
logger.setLevel(logging.ERROR)

print("----- Starting production server with Waitress -----")
print("----- Access logs are disabled. Server is running at http://0.0.0.0:5000 -----")
serve(app, host='0.0.0.0', port=5000)
