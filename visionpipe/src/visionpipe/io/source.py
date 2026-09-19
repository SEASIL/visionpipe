"""Frame ingestion for files, webcams and RTSP/HTTP streams.

Design notes (these matter for live video):
* Live streams are read on a background thread into a tiny queue that DROPS THE OLDEST frame
  when full. If inference is slower than the camera, the pipeline always sees the freshest frame
  instead of falling further and further behind (unbounded latency).
* The reader reconnects with exponential backoff when a camera drops.
* File sources are read synchronously and deterministically, with timestamps = index / fps,
  so that results (and unit tests) are reproducible.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Iterator, Optional, Union

import cv2

from ..types import Frame

log = logging.getLogger(__name__)

Uri = Union[str, int]
_LIVE_PREFIXES = ("rtsp://", "rtmp://", "udp://", "http://", "https://")


def is_live_uri(uri: Uri) -> bool:
    return isinstance(uri, int) or str(uri).lower().startswith(_LIVE_PREFIXES)


class VideoSource:
    def __init__(
        self,
        uri: Uri,
        camera_id: str = "cam0",
        loop: bool = False,
        reconnect: bool = True,
        max_queue: int = 2,
        live: Optional[bool] = None,
        max_backoff_s: float = 30.0,
    ):
        self.uri = uri
        self.camera_id = camera_id
        self.loop = loop
        self.reconnect = reconnect
        self.max_queue = max_queue
        self.live = is_live_uri(uri) if live is None else live
        self.max_backoff_s = max_backoff_s
        self.fps: float = 25.0
        self.size: Optional[tuple] = None  # (w, h) once known
        self._stop = threading.Event()

    # ---------------------------------------------------------------- public
    def frames(self) -> Iterator[Frame]:
        yield from (self._live_frames() if self.live else self._file_frames())

    def close(self) -> None:
        self._stop.set()

    # ---------------------------------------------------------------- internals
    def _open(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.uri)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return cap

    def _file_frames(self) -> Iterator[Frame]:
        cap = self._open()
        if cap is None:
            raise FileNotFoundError(f"Cannot open video source: {self.uri}")
        idx = 0
        try:
            while not self._stop.is_set():
                ok, img = cap.read()
                if not ok:
                    if self.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                yield Frame(self.camera_id, idx, idx / self.fps, img)
                idx += 1
        finally:
            cap.release()

    def _live_frames(self) -> Iterator[Frame]:
        q: "queue.Queue" = queue.Queue(maxsize=self.max_queue)

        def put_drop_oldest(item) -> None:
            while True:
                try:
                    q.put_nowait(item)
                    return
                except queue.Full:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass

        def reader() -> None:
            backoff = 1.0
            while not self._stop.is_set():
                cap = self._open()
                if cap is None:
                    if not self.reconnect:
                        break
                    log.warning("[%s] cannot open %s, retrying in %.0fs", self.camera_id, self.uri, backoff)
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, self.max_backoff_s)
                    continue
                backoff = 1.0
                while not self._stop.is_set():
                    ok, img = cap.read()
                    if not ok:
                        break
                    put_drop_oldest((time.time(), img))
                cap.release()
                if not self.reconnect:
                    break
                log.warning("[%s] stream lost, reconnecting", self.camera_id)
            put_drop_oldest(None)  # sentinel: reader finished

        t = threading.Thread(target=reader, name=f"reader-{self.camera_id}", daemon=True)
        t.start()
        idx = 0
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                ts, img = item
                yield Frame(self.camera_id, idx, ts, img)
                idx += 1
        finally:
            self._stop.set()
