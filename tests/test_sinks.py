import time

from visionpipe.events.sinks import EventSink
from visionpipe.types import Event


class DummySink(EventSink):
    def __init__(self, **kwargs):
        super().__init__("dummy", **kwargs)
        self.processed = []

    def process(self, event: Event) -> None:
        self.processed.append(event)


def test_sink_processes_events_asynchronously():
    sink = DummySink()
    e1 = Event(type="test", camera_id="c1", frame_index=1, timestamp=0.1, zone="z1")
    e2 = Event(type="test2", camera_id="c1", frame_index=2, timestamp=0.2, zone="z2")
    
    sink.send(e1)
    sink.send(e2)
    
    # Wait for the background thread to process
    sink.close()
    
    assert len(sink.processed) == 2
    assert sink.processed[0] == e1
    assert sink.processed[1] == e2


def test_sink_drops_events_when_queue_full():
    sink = DummySink(maxsize=2)
    # block the worker
    def slow_process(e):
        time.sleep(0.5)
    sink.process = slow_process
    
    for i in range(10):
        sink.send(Event(type="test", camera_id="c1", frame_index=i, timestamp=0.1, zone="z"))
    
    # The first one is taken by process() immediately (so queue has 0).
    # Then 2 go into the queue. The rest (7) get dropped.
    assert sink.queue.full() or sink.queue.qsize() > 0
    sink.close()
