"""Callback Handler to capture events in a Queue and provide a Generator to iterate over them."""

import logging
import queue
import time
from collections.abc import Generator
from typing import Any, Optional

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEvent, CBEventType

logger = logging


class CBEventStart(CBEvent):  # pylint: disable=too-few-public-methods
    """Used to mark the start of an event."""


class CBEventEnd(CBEvent):  # pylint: disable=too-few-public-methods
    """Used to mark the end of an event."""


class EventStreamer(BaseCallbackHandler):
    """Callback handler for logging events and streaming them at each step of
    the RAG chat flow.
    """

    _queue: queue.Queue[CBEvent]
    _verbose: bool
    _is_trace_over: bool
    _is_done: bool
    _event_counter: int  # incr on start events, decr on end events

    def __init__(
        self, verbose: bool = False, ignored_events: list[CBEventType] | None = None
    ) -> None:
        if ignored_events is None:
            ignored_events = []
        super().__init__(
            event_starts_to_ignore=ignored_events, event_ends_to_ignore=ignored_events
        )
        self._queue = queue.Queue()

        self._event_counter = 0

        self._is_done = False
        self._is_trace_over = False
        self._verbose = verbose

    @property
    def event_queue(self) -> queue.Queue[CBEvent]:
        """Return the event queue."""
        return self._queue

    @property
    def is_stream_done(self) -> bool:
        """Return True if the event stream is done."""
        return self._is_done

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        if self._verbose:
            logger.debug(f"start_trace {trace_id}")
        self._is_done = False
        self._is_trace_over = False

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[dict[str, list[str]]] = None,
    ) -> None:
        self._is_trace_over = True
        if self._event_counter == 0:
            self._is_done = True
        if self._verbose:
            logger.debug(
                f"end_trace {trace_id} _event_counter {self._event_counter} _is_trace_over {self._is_trace_over} _is_done {self._is_done}"
            )

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Add event to queue & return event id."""
        now: str = str(time.time())
        event = CBEventStart(event_type, payload=payload, time=now, id_=event_id)
        self._queue.put_nowait(event)
        self._event_counter += 1
        if self._verbose:
            logger.debug(
                f"Added to Queue: {event_type} START. id: {event_id} _event_counter: {self._event_counter} _is_trace_over: {self._is_trace_over} _is_done: {self._is_done}"
            )
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Add event to queue & return event id."""
        now: str = str(time.time())
        event = CBEventEnd(event_type, payload=payload, time=now, id_=event_id)
        self._event_counter -= 1
        self._queue.put_nowait(event)
        if self._is_trace_over and self._event_counter == 0:
            self._is_done = True
        if self._verbose:
            logger.debug(
                f"Added to Queue: {event_type} END. id: {event_id} _event_counter: {self._event_counter} _is_trace_over: {self._is_trace_over} _is_done: {self._is_done}"
            )

    @property
    def event_gen(self) -> Generator[CBEvent, None, None]:
        while not self._is_done or not self._queue.empty():
            try:
                event: CBEvent = self._queue.get(block=False)
                yield event
            except queue.Empty:
                # Queue is empty, but we're not done yet. Sleep for 0secs to release the GIL.
                time.sleep(0)
