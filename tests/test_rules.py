from tests.helpers import frame, make_track
from visionpipe.events import (
    AbandonedObject,
    CrowdDetection,
    EventEngine,
    LineCrossing,
    Loitering,
    ZoneIntrusion,
    build_rules,
)
from visionpipe.events.rules import Context

ZONE = [(100, 100), (300, 100), (300, 300), (100, 300)]


def ctx(i, tracks, fps=25.0):
    return Context("cam0", i, i / fps, tracks)


# ------------------------------------------------------------------ intrusion
def test_intrusion_requires_persistence_and_fires_once():
    rule = ZoneIntrusion("restricted", ZONE, classes=["person"], min_frames=5)
    events = []
    for i in range(20):
        events += rule.update(ctx(i, [make_track(1, 200, 200)]))  # foot at y=240 -> inside
    assert len(events) == 1
    assert events[0].type == "zone_intrusion" and events[0].frame_index == 4 and events[0].zone == "restricted"


def test_intrusion_ignores_flicker_and_wrong_class():
    rule = ZoneIntrusion("restricted", ZONE, classes=["person"], min_frames=5)
    events = []
    for i in range(3):  # only 3 frames inside: below persistence threshold
        events += rule.update(ctx(i, [make_track(1, 200, 200)]))
    for i in range(3, 30):
        events += rule.update(ctx(i, [make_track(1, 500, 500)]))
    for i in range(30, 60):
        events += rule.update(ctx(i, [make_track(2, 200, 200, cls="car")]))
    assert events == []


def test_intrusion_refires_after_real_exit():
    rule = ZoneIntrusion("z", ZONE, min_frames=3, exit_frames=5)
    events = []
    seq = [(200, 200)] * 10 + [(600, 600)] * 10 + [(200, 200)] * 10
    for i, (x, y) in enumerate(seq):
        events += rule.update(ctx(i, [make_track(1, x, y)]))
    assert len(events) == 2


def test_intrusion_short_exit_does_not_refire():
    rule = ZoneIntrusion("z", ZONE, min_frames=3, exit_frames=15)
    events = []
    seq = [(200, 200)] * 10 + [(600, 600)] * 4 + [(200, 200)] * 10  # 4-frame jitter outside
    for i, (x, y) in enumerate(seq):
        events += rule.update(ctx(i, [make_track(1, x, y)]))
    assert len(events) == 1


# ------------------------------------------------------------------ loitering
def test_loitering_fires_after_dwell_time():
    rule = Loitering("lobby", ZONE, dwell_seconds=4.0)
    events = []
    for i in range(200):  # 8 s at 25 fps
        events += rule.update(ctx(i, [make_track(1, 200, 200)]))
    assert len(events) == 1
    assert 4.0 <= events[0].details["dwell_seconds"] <= 4.1


def test_loitering_walk_through_does_not_fire():
    rule = Loitering("lobby", ZONE, dwell_seconds=4.0, exit_frames=5)
    events = []
    for i in range(60):  # crosses zone in ~1.6 s then leaves
        events += rule.update(ctx(i, [make_track(1, 50 + 8 * i, 200)]))
    assert events == []


# ------------------------------------------------------------------ line crossing
def test_line_crossing_counts_direction():
    rule = LineCrossing("door", (0, 200), (400, 200), min_gap_frames=5)
    events = []
    for i in range(20):  # moves down through the line
        events += rule.update(ctx(i, [make_track(1, 200, 100 + 10 * i)]))
    for i in range(20, 40):  # then back up
        events += rule.update(ctx(i, [make_track(1, 200, 100 + 10 * (39 - i))]))
    dirs = [e.details["direction"] for e in events]
    assert len(dirs) == 2 and set(dirs) == {"forward", "backward"}
    assert rule.counts == {"forward": 1, "backward": 1}


def test_line_crossing_ignores_passing_beside_segment():
    rule = LineCrossing("door", (0, 200), (100, 200))  # short segment
    events = []
    for i in range(30):  # crosses the *infinite* line at x=300, outside the segment
        events += rule.update(ctx(i, [make_track(1, 300, 100 + 10 * i)]))
    assert events == []


def test_line_crossing_jitter_is_debounced():
    rule = LineCrossing("door", (0, 200), (400, 200), min_gap_frames=20)
    events = []
    for i in range(20):  # oscillates across the line every frame, inside the 20-frame gap
        y = 195 if i % 2 == 0 else 205
        events += rule.update(ctx(i, [make_track(1, 200, y, h=0.001)]))
    assert len(events) == 1


def test_line_direction_filter():
    rule = LineCrossing("door", (0, 200), (400, 200), direction="forward", min_gap_frames=1)
    events = []
    for i in range(20):
        events += rule.update(ctx(i, [make_track(1, 200, 100 + 10 * i)]))
    for i in range(20, 40):
        events += rule.update(ctx(i, [make_track(1, 200, 100 + 10 * (39 - i))]))
    assert len(events) <= 1


# ------------------------------------------------------------------ crowd
def test_crowd_fires_with_hysteresis():
    rule = CrowdDetection("plaza", ZONE, threshold=3, min_frames=5, clear_frames=10)
    people = [make_track(i, 150 + 30 * i, 200) for i in range(4)]
    events = []
    for i in range(20):
        events += rule.update(ctx(i, people))
    assert [e.type for e in events] == ["crowd_detected"]
    assert events[0].details["count"] == 4
    for i in range(20, 25):  # brief dip below threshold: no clear yet
        events += rule.update(ctx(i, people[:2]))
    assert [e.type for e in events] == ["crowd_detected"]
    for i in range(25, 45):
        events += rule.update(ctx(i, people[:2]))
    assert [e.type for e in events] == ["crowd_detected", "crowd_cleared"]


# ------------------------------------------------------------------ abandoned object
def test_abandoned_object_fires_when_owner_leaves():
    rule = AbandonedObject("hall", stationary_seconds=5.0, owner_radius_px=100)
    bag = make_track(10, 300, 300, w=30, h=30, cls="backpack")
    events = []
    for i in range(50):  # owner beside the bag for 2 s
        events += rule.update(ctx(i, [bag, make_track(1, 320, 300)]))
    assert events == []
    for i in range(50, 250):  # owner walked away: 8 s alone
        events += rule.update(ctx(i, [bag, make_track(1, 800, 300)]))
    assert len(events) == 1 and events[0].type == "abandoned_object"


def test_object_with_owner_nearby_never_fires():
    rule = AbandonedObject("hall", stationary_seconds=5.0, owner_radius_px=100)
    bag = make_track(10, 300, 300, w=30, h=30, cls="backpack")
    events = []
    for i in range(500):
        events += rule.update(ctx(i, [bag, make_track(1, 330, 300)]))
    assert events == []


def test_moving_object_is_not_abandoned():
    rule = AbandonedObject("hall", stationary_seconds=2.0)
    events = []
    for i in range(300):
        bag = make_track(10, 300 + 3 * i, 300, w=30, h=30, cls="backpack")
        events += rule.update(ctx(i, [bag]))
    assert events == []


# ------------------------------------------------------------------ engine / config
def test_engine_and_build_rules_from_config():
    rules = build_rules(
        [
            {"type": "zone_intrusion", "name": "a", "polygon": ZONE, "min_frames": 2},
            {"type": "line_crossing", "name": "b", "p1": [0, 200], "p2": [400, 200]},
        ]
    )
    engine = EventEngine(rules)
    events = []
    for i in range(10):
        events += engine.update(frame(i), [make_track(1, 200, 100 + 12 * i)])
    kinds = {e.type for e in events}
    assert {"zone_intrusion", "line_crossing"} <= kinds


def test_build_rules_rejects_unknown_type():
    import pytest

    with pytest.raises(ValueError):
        build_rules([{"type": "teleport", "name": "x"}])


def test_stale_state_is_garbage_collected():
    rule = ZoneIntrusion("z", ZONE, min_frames=2)
    for i in range(5):
        rule.update(ctx(i, [make_track(1, 200, 200)]))
    assert 1 in rule._state
    rule.update(ctx(1000, []))
    assert rule._state == {}
