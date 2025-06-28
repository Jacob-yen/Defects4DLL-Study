import logging
import os
import datetime
import configparser


class LoggerUtils:
    instance = None

    def __init__(self,log_dir=None,log_name=None):
        self.log_dir = log_dir
        self.log_name = log_name
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        self.logger = self.setup_logger()

    def setup_logger(self):
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s - %(filename)s: line-%(lineno)d - %(levelname)s]: %(message)s')

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if self.log_name and self.log_dir:
            log_filename = f"log_{self.log_name}_{formatted_time}.log"
            log_path = os.path.join(self.log_dir, log_filename)
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    @classmethod
    def get_instance(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = cls(*args, **kwargs)
        return cls.instance
