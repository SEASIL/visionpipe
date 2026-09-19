# Failure analysis (fill in from YOUR footage)

For each condition: reproduce it on a short clip, measure the impact, apply a fix, re-measure.
Keep the before/after numbers. This document is what shows you can debug vision systems in the real world.

| Condition | Symptom observed | Root cause | Fix applied | Before → after (metric) |
|---|---|---|---|---|
| Low light / night | _e.g. missed detections, mAP drop_ | _noise, low contrast_ | _brightness augmentation (`hsv_v`), add night data, CLAHE_ | _mAP@0.5 A → B_ |
| Partial occlusion | _ID switches when people cross_ | _IoU-only association_ | _lower `match_iou`, appearance features, raise `max_lost`_ | _ID switches X → Y_ |
| Camera angle / small objects | _far people missed_ | _objects below ~20 px_ | _higher `imgsz`, tiled inference, fine-tune on aerial-style data_ | _recall A → B_ |
| Motion blur / compression | _confidence flicker, fragmented tracks_ | _low-confidence detections_ | _ByteTrack low-score stage, `min_hits`_ | _fragmentations X → Y_ |
| Zone-edge jitter | _repeated intrusion alerts_ | _foot point oscillating across the polygon edge_ | _`exit_frames` debounce, fire-once_ | _alerts/hour X → Y_ |
| Stream drops | _pipeline stalls_ | _RTSP disconnect_ | _auto-reconnect with backoff_ | _recovery time_ |

Ablations worth running: event rules with and without persistence/debounce (count false alerts on the same clip),
tracker with and without the low-confidence stage (IDF1, ID switches).
