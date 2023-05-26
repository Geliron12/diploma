import logging

loggers_dict = {}

def create_logger(name:str):
    global loggers_dict
    if name in loggers_dict:
        return loggers_dict[name]
    else:  
        logger = logging.getLogger(name)
        loggers_dict[name] = logger
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger

def set_log_level(name, level):
    logger_names = []
    if name == "all":
        logger_names = list(loggers_dict.keys())
    else:
        logger_names = [name]
    for name in logger_names:
        logger = loggers_dict[name]
        logger.setLevel(level)
