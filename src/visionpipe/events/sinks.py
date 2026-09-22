import json
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

from ..types import Event

log = logging.getLogger(__name__)


class EventSink(ABC):
    """Base class for asynchronous event sinks with a bounded queue."""

    def __init__(self, name: str, maxsize: int = 1000):
        self.name = name
        self.queue: queue.Queue[Optional[Event]] = queue.Queue(maxsize=maxsize)
        self.thread = threading.Thread(target=self._worker, daemon=True, name=f"sink_{name}")
        self.thread.start()

    def send(self, event: Event) -> None:
        """Push an event to the sink. If the queue is full, drop it so pipeline doesn't block."""
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            log.warning("[%s] Event queue full. Dropped event: %s", self.name, event.type)

    def close(self) -> None:
        self.queue.put(None)  # Sentinel to stop thread
        self.thread.join(timeout=2.0)

    def _worker(self) -> None:
        while True:
            event = self.queue.get()
            if event is None:
                break
            try:
                self.process(event)
            except Exception as e:
                log.error("[%s] Error processing event: %s", self.name, e)
            finally:
                self.queue.task_done()

    @abstractmethod
    def process(self, event: Event) -> None: ...


class WebhookSink(EventSink):
    """Sends events via HTTP POST to a webhook URL with retries."""

    def __init__(self, url: str, retries: int = 3, **kwargs):
        super().__init__("webhook", **kwargs)
        self.url = url
        self.retries = retries
        try:
            import requests

            self.requests = requests
        except ImportError:
            log.warning("requests package not installed. WebhookSink will fail.")
            self.requests = None

    def process(self, event: Event) -> None:
        if self.requests is None:
            return
        payload = event.to_dict()
        for attempt in range(self.retries):
            try:
                resp = self.requests.post(self.url, json=payload, timeout=5.0)
                resp.raise_for_status()
                return
            except Exception as e:
                if attempt == self.retries - 1:
                    log.error("[webhook] Failed to post event to %s: %s", self.url, e)
                else:
                    time.sleep(1.0)


class MQTTSink(EventSink):
    """Publishes events to an MQTT broker."""

    def __init__(self, host: str, port: int = 1883, topic: str = "visionpipe/events", **kwargs):
        super().__init__("mqtt", **kwargs)
        self.topic = topic
        try:
            import paho.mqtt.publish as publish

            self.publish = publish
        except ImportError:
            log.warning("paho-mqtt package not installed. MQTTSink will fail.")
            self.publish = None
        self.host = host
        self.port = port

    def process(self, event: Event) -> None:
        if self.publish is None:
            return
        payload = json.dumps(event.to_dict())
        try:
            self.publish.single(self.topic, payload=payload, hostname=self.host, port=self.port, retain=False)
        except Exception as e:
            log.error("[mqtt] Failed to publish event: %s", e)
