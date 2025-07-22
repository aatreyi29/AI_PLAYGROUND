# import logging
# import os
# from datetime import datetime

# LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
# logs_path = os.path.join(os.getcwd(), 'logs',LOG_FILE)
# os.makedirs(logs_path,exist_ok=True)

# LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

# logging.basicConfig(
#     filename=LOG_FILE_PATH,
#     format='[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
#     level=logging.INFO)

import os
import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

# Log Directory Setup
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create log file with dynamic timestamp
file_format = f"{datetime.now().strftime('%d_%m_%y_%H_%M_%S')}.log"
LOG_FILE = os.path.join(LOG_DIR, file_format)

# Logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set default logging level to DEBUG

# File Handler - Timed Rotating
file_handler = TimedRotatingFileHandler(
    LOG_FILE, 
    when="midnight",  # Log file rotation happens at midnight
    interval=1,       # Interval of 1 day
    backupCount=7,    # Keep 7 backup logs
    encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)

# Formatter to output the log messages in a readable format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Console Handler - for printing to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# You can now import 'logger' from this file in other modules
logger = logging.getLogger("ufresources-ai")
logger.setLevel(logging.INFO)

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter and add to handler
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
ch.setFormatter(formatter)

# Add handler to logger if not already added
if not logger.hasHandlers():
    logger.addHandler(ch)
