#  Copyright 2025 Amazon.com, Inc. or its affiliates.

from unittest import TestCase

from aws.osml.model_runner.common.observable_event import ObservableEvent


class TestObservableEvent(TestCase):
    def test_subscribe_handler(self):
        """Test that handlers can be subscribed to the event."""
        event = ObservableEvent()
        mock_handler_called = False

        def mock_handler():
            nonlocal mock_handler_called
            mock_handler_called = True

        event.subscribe(mock_handler)
        event()
        self.assertTrue(mock_handler_called)

    def test_unsubscribe_handler(self):
        """Test that handlers can be unsubscribed from the event."""
        event = ObservableEvent()
        mock_handler_called = False

        def mock_handler():
            nonlocal mock_handler_called
            mock_handler_called = True

        event.subscribe(mock_handler)
        event.unsubscribe(mock_handler)
        event()
        self.assertFalse(mock_handler_called)

    def test_multiple_handlers(self):
        """Test that multiple handlers can be subscribed and all get called."""
        event = ObservableEvent()
        call_count = 0

        def handler1():
            nonlocal call_count
            call_count += 1

        def handler2():
            nonlocal call_count
            call_count += 1

        event.subscribe(handler1)
        event.subscribe(handler2)
        event()
        self.assertEqual(call_count, 2)

    def test_handler_with_arguments(self):
        """Test that handlers receive the correct arguments."""
        event = ObservableEvent()
        received_args = None
        received_kwargs = None

        def handler(*args, **kwargs):
            nonlocal received_args, received_kwargs
            received_args = args
            received_kwargs = kwargs

        event.subscribe(handler)
        event("test", 123, keyword="value")
        self.assertEqual(received_args, ("test", 123))
        self.assertEqual(received_kwargs, {"keyword": "value"})

    def test_unsubscribe_nonexistent_handler(self):
        """Test that unsubscribing a non-existent handler raises ValueError."""
        event = ObservableEvent()

        def handler():
            pass

        with self.assertRaises(ValueError):
            event.unsubscribe(handler)

    def test_handler_execution_order(self):
        """Test that handlers are executed in the order they were subscribed."""
        event = ObservableEvent()
        execution_order = []

        def handler1():
            execution_order.append(1)

        def handler2():
            execution_order.append(2)

        def handler3():
            execution_order.append(3)

        event.subscribe(handler1)
        event.subscribe(handler2)
        event.subscribe(handler3)
        event()
        self.assertEqual(execution_order, [1, 2, 3])

    def test_handler_exception_handling(self):
        """
        Test that an exception in the first handler will be caught and not prevent other handlers
        from executing.
        """
        event = ObservableEvent()
        second_handler_called = False

        def failing_handler():
            raise Exception("Handler failed")

        def second_handler():
            nonlocal second_handler_called
            second_handler_called = True

        event.subscribe(failing_handler)
        event.subscribe(second_handler)

        event()
        self.assertTrue(second_handler_called)

    def test_subscribe_same_handler_multiple_times(self):
        """Test that the same handler can be subscribed multiple times."""
        event = ObservableEvent()
        call_count = 0

        def handler():
            nonlocal call_count
            call_count += 1

        event.subscribe(handler)
        event.subscribe(handler)
        event()
        self.assertEqual(call_count, 2)

    def test_lambda_handler(self):
        """Test that lambda functions can be used as handlers."""
        event = ObservableEvent()
        result = []

        event.subscribe(lambda x: result.append(x))
        event("test")
        self.assertEqual(result, ["test"])

    def test_method_handler(self):
        """Test that class methods can be used as handlers."""

        class TestClass:
            def __init__(self):
                self.was_called = False

            def handler(self):
                self.was_called = True

        test_obj = TestClass()
        event = ObservableEvent()

        event.subscribe(test_obj.handler)
        event()
        self.assertTrue(test_obj.was_called)
