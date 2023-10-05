from data_processing_unit import Preprocessor
from config import CONFIG
import time
import os


TEST_MODE = False


if __name__ == "__main__":
    path = CONFIG.storage + "/" + CONFIG.version
    if not os.path.exists(path):
        os.makedirs(path)
    start = time.time()

    data_preprocessor = Preprocessor(test_mode=TEST_MODE)
    data_preprocessor.run_modin()

    duration = time.time() - start
    print(f"Time : {duration}")
