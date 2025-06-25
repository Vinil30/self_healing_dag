import logging
from datetime import datetime
import json

def setup_logging():
    logger = logging.getLogger("self_healing_dag")
    logger.setLevel(logging.INFO)
    
    # File handler
    log_file = f"logs/classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    
    logger.addHandler(file_handler)
    return logger