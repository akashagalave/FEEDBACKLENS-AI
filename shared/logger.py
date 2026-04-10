import sys
from loguru import logger


def get_logger(service_name: str):
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> | <bold>{message}</bold>",
        level="INFO"
    )
    logger.add(
        f"reports/{service_name}.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG"
    )
    return logger.bind(service=service_name)