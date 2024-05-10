import logging
import os


def get_logger(logger_name):
    log_file_path = os.path.join(os.getcwd(), "logs", logger_name)
    # 配置日志输出的格式
    logging.basicConfig(
        level=logging.INFO,  # 日志级别为 INFO
        format="%(asctime)s [%(levelname)s] %(message)s",  # 日志格式
        handlers=[
            logging.FileHandler(log_file_path+'.log'),  # 将日志写入文件
            logging.StreamHandler()  # 将日志打印到控制台
        ]
    )

    # 创建一个日志对象
    logger = logging.getLogger(logger_name)
    return logger

if __name__ == "__main__":
    # 创建一个日志对象
    logger = get_logger("my_logger")

    # 使用日志对象记录日志
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")