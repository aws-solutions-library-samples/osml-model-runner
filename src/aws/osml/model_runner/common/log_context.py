import logging
import threading
from typing import List, Optional

_LOG_CONTEXT = threading.local()


class ThreadingLocalContextFilter(logging.Filter):
    """
    This is a filter that injects contextual information into the log message. The contextual information is
    set using the static methods of this class.
    """

    def __init__(self, attribute_names: List[str]):
        super().__init__()
        self.attribute_names = attribute_names

    def filter(self, record: logging.LogRecord) -> bool:
        """
        This method is called for each log record. It injects the contextual information into the log record.

        :param record: the log record to filter
        :return: True, this filter does not exclude information from the log
        """
        for attribute_name in self.attribute_names:
            setattr(record, attribute_name, getattr(_LOG_CONTEXT, attribute_name, None))
        return True

    @staticmethod
    def set_context(context: Optional[dict]):
        """
        Set the context for the current thread. If None all context information is cleared.

        :param context: dict = the context to set
        """
        if context is None:
            _LOG_CONTEXT.__dict__.clear()
        else:
            _LOG_CONTEXT.__dict__.update(context)
