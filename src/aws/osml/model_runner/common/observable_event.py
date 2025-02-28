#  Copyright 2025 Amazon.com, Inc. or its affiliates.

import logging
from typing import Any, Callable, List

# Enable a cleaner type alias once we drop support for Python 3.9
# ObservableEventHandler: TypeAlias = Callable[..., Any]

logger = logging.getLogger(__name__)


class ObservableEvent:
    """
    A class that allows subscribers to be notified of observable events.

    A handler is simply a callable that can be a function, lambda, or class method implemented by a subscriber. It
    must accept positional and keyword arguments that match the parameters used when the event is triggered.

    In general handlers should not raise exceptions. This would be poor programming practice because it would allow
    a subscriber to inject new exceptions into the execution path of the event provider. In cases where a handler
    ignores this guidance and raises an exception this implementation will log and swallow the exception.

    If multiple handlers are subscribed to an event they will be executed in the order in which they were subscribed.
    If the same handler is subscribed more than once it will be executed multiple times.

    Example:
        class MyClass:
            def __init__(self):
                self.my_event = ObservableEvent()

            def do_something(self):
                # Perform some action
                print("Something happened!")
                self.my_event() # Trigger the event

        def my_handler():
            print("Event was triggered!")

        # Usage
        obj = MyClass()
        obj.my_event.subscribe(my_handler) # Subscribe to the event
        obj.do_something() # Output: Something happened!\nEvent was triggered!
        obj.my_event.unsubscribe(my_handler) # Unsubscribe from the event
        obj.do_something() # Output: Something happened!

    """

    def __init__(self):
        """
        Initialize a new ObservableEvent instance with an empty list of handlers.
        """
        self._handlers: List[Callable[..., Any]] = []

    def subscribe(self, handler: Callable[..., Any]) -> None:
        """
        Register a handler function for this event. The handler should be callable with whatever arguments are
        typically fired with the event.

        :param handler: the function to invoke when the event occurs
        """
        self._handlers.append(handler)

    def unsubscribe(self, handler: Callable[..., Any]) -> None:
        """
        Remove a previously registered handler function from this event.

        :param handler: The handler function to remove from the event's notification list.
        :raises ValueError: If the handler is not found in the list of registered handlers.
        """
        self._handlers.remove(handler)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """
        This method is called when the event occurs to notify all handlers.

        When the ObservableEvent instance is called like a function, all registered handlers
        will be invoked in the order they were added, with the same arguments that were passed
        to this method.

        :param args: Variable positional arguments that will be passed to each handler.
        :param kwargs: Variable keyword arguments that will be passed to each handler.
        """
        for handler in self._handlers:
            try:
                handler(*args, **kwargs)
            except Exception as e:
                logger.error("Exception raised by event handler.", exc_info=True)
                logger.error(f"Exception: {e} has been ignored because it was raised by an event handler.")
