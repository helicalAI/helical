import os
import logging

logging.captureWarnings(True)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logger = logging.getLogger('helical')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.propagate = False  # Prevent propagation to root

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
