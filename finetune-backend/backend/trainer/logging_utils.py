from rich.console import Console
from rich.logging import RichHandler
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=Console(), omit_repeated_times=False)]
    )
