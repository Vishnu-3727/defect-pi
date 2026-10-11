# 16 — The Flask Dashboard

**Prerequisites:** [15 — SQLite and result storage](15_sqlite.md).

---

## 1. Concept

Flask is a minimal Python web framework. You write a function, decorate it with
a URL, and it becomes a web page:

```python
@app.route("/api/stats")
def stats():
    return jsonify(pipeline.stats())
```

That is essentially the whole framework. It provides routing, request parsing
and templating, and deliberately provides no database layer, no authentication
and no admin panel.

Our dashboard is one process doing two jobs: running the inspection pipeline in
a background thread, and serving a live video stream plus statistics over HTTP.

---

## 2. Why it is needed

A headless Pi in an enclosure has no screen. Without a dashboard, the only way
to know whether inspection is working is to SSH in and read logs — which no
operator will do.

Three audiences, three needs:

**The operator** needs to see that the camera is aimed correctly, the lighting
has not changed, and parts are being classified sensibly. That requires a *live
image with boxes drawn on it*, not a number.

**The supervisor** needs the reject rate and its trend.

**You, at Review 2** need to show the system working to people standing behind
you, from a projector. A terminal full of log lines does not demonstrate a
vision system; a live annotated video feed does.

---

## 3. Real-world example

Every industrial machine has an HMI — Human-Machine Interface — traditionally a
dedicated touch panel wired to a PLC, costing more than this entire project.

Modern edge systems replace it with a web server on the device. The advantage is
that any phone, laptop or plant PC on the network becomes the HMI with no
software to install. This is the standard architecture for machine-vision
appliances from Cognex, Keyence and similar vendors, and it is exactly what we
are building at student scale.

---

## 4. How it works

### The architecture

```
                 +-----------------------------+
                 |   dashboard/app.py process  |
                 |                             |
 camera ---->    |  [pipeline thread]          |
                 |    capture -> detect ->     |
                 |    decide -> store          |
                 |         |                   |
                 |         v                   |
                 |   latest_jpeg (shared)      |
                 |   inspections.db            |
                 |         |                   |
                 |  [Flask threads]            |
                 |    /  /stream  /api/*       |
                 +-----------------------------+
                            |
                     HTTP to browsers
```

One process. The pipeline runs in a daemon thread started by
`pipeline.run_in_background()`; Flask serves requests in its own threads.

### The four routes

| Route | Returns |
|---|---|
| `/` | the HTML page |
| `/stream` | `multipart/x-mixed-replace` MJPEG video |
| `/api/stats` | JSON counters, live and from the database |
| `/api/recent` | JSON list of recent inspections |

### MJPEG streaming

There is no video codec here. **MJPEG is a sequence of complete JPEG images sent
down one never-ending HTTP response**, separated by a boundary marker:

```
--frame
Content-Type: image/jpeg

<jpeg bytes>
--frame
Content-Type: image/jpeg

<jpeg bytes>
...
```

The response never ends. The browser paints each image as it arrives, and
`<img src="/stream">` displays live video with no JavaScript, no WebSocket and
no player library.

It is bandwidth-inefficient — every frame is a keyframe, no inter-frame
compression — and it is the right choice here: zero dependencies, zero latency,
works in every browser, and trivially debuggable with `curl`.

In Flask it is a generator:

```python
def frames():
    while True:
        jpeg = pipeline.latest_jpeg()
        if jpeg is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(1.0 / STREAM_FPS)

@app.route("/stream")
def stream():
    return Response(frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
```

### Decoupling: the single most important design decision

The stream generator **never asks the pipeline for a frame**. It reads whatever
the pipeline last published.

```python
# in the pipeline, after annotating
ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
if ok:
    self._latest_jpeg = buf.tobytes()   # atomic rebind
```

If the browser is slow, or three people open the page, or someone's WiFi drops
mid-stream, the pipeline is unaffected. The viewer simply misses frames.

The alternative — a queue the pipeline pushes into — creates **back-pressure**:
when the consumer stalls, the queue fills and the producer blocks. Inspection
would then run at the speed of the slowest browser on the network. For a system
that ejects physical parts, that is a safety problem, not a performance one.

A single Python attribute rebind is atomic under the GIL, so no lock is needed:
a reader gets either the old bytes object or the new one, never a mixture.

### Rate limiting the stream

`STREAM_FPS = 15`, while the pipeline runs faster. Streaming every frame would
spend CPU on JPEG encoding and network on bytes that no human eye can resolve.
15 FPS looks smooth; the saved cycles go to inference.

---

## 5. Mathematical intuition

### Bandwidth

A 640×480 JPEG at quality 80 is roughly 40 KB.

$$40\,\text{KB} \times 15\,\text{fps} = 600\,\text{KB/s} \approx 4.8\ \text{Mbit/s}$$

Fine on Ethernet or good WiFi, per viewer. Three viewers is ~14 Mbit/s, which is
why the design assumes one or two operators, not a public page.

Compare H.264 at the same resolution and quality: roughly 1 Mbit/s, a 5×
saving — bought with an encoder dependency, a player, and buffering latency.
**For a control interface, latency matters more than bandwidth.**

### Why JPEG encoding is not free

Encoding costs a few milliseconds per frame on the Pi. At 36 FPS that would be
meaningful; at 15 FPS it is around 5% of one core. The rate limit is what keeps
this negligible — this is the arithmetic behind that constant.

---

## 6. Software implementation

`dashboard/app.py`.

```python
app.run(host=cfg.dashboard.host, port=cfg.dashboard.port,
        threaded=True, use_reloader=False)
```

Both keyword arguments are load-bearing.

**`threaded=True`** — without it, Flask's development server handles one request
at a time. `/stream` is an infinite response, so it would occupy the single
handler forever and `/api/stats` would never answer. The page would show live
video and permanently frozen numbers.

**`use_reloader=False`** — Flask's auto-reloader runs your application **twice**:
a parent watcher and a child that serves. With a pipeline started at import
time, that means two pipelines, two cameras contending for one device, two
writers to one database, and doubled CPU. The symptom is bizarre — halved frame
rate and duplicated rows — and the cause is one default argument.

### Statistics: live and durable

```json
{
  "live":     { "frames": 139, "passes": 15, "fails": 124, "fps": 16.8 },
  "database": { "total": 139, "reject_rate": 89.2, "by_class": {...} },
  "config":   { "actuator": "mock", "conf_threshold": 0.35 }
}
```

`live` comes from in-memory counters since the process started; `database`
aggregates all history. Both are shown because they answer different questions —
"is it working right now?" versus "what has it done overall?" A dashboard that
conflates them hides restarts.

`config` is included so a screenshot is self-describing: months later you can
see which threshold produced those numbers.

---

## 7. Hardware implementation

**CPU contention is the real hardware issue.** Flask, JPEG encoding and
inference share four cores. `detector.num_threads: 4` already claims all of
them, so serving the dashboard steals from inference — visible in the benchmark
as the p95 tail at four threads. Stop the service before benchmarking, and treat
2 threads as the option if a latency bound is ever required.

**Network access.** The service binds `0.0.0.0:5000`, reachable from any machine
on the LAN. For a demo, connect the laptop and Pi to the same network, or run a
direct Ethernet cable — link-local addressing works with no router.

**No authentication.** Anyone on the network can view the stream and the
statistics. Acceptable for a lab and a college demo; unacceptable on a factory
network. The fix is a reverse proxy with HTTP basic auth, not application code.

**The Flask development server is not a production server.** It is
single-process and has no request limits. For deployment the answer is
`waitress` or `gunicorn` — one line of change. It is not done here because the
constraint is honest: this is a demonstrator.

---

## 8. Common mistakes

**Running the pipeline inside a request handler.** Inspection would then only
happen while someone is watching the page.

**Letting the stream drive the pipeline.** Back-pressure: the line runs at
browser speed. Publish-latest, never a queue.

**Leaving `use_reloader=True`.** Two pipelines, one camera, doubled rows. Costs
hours to diagnose.

**Forgetting `threaded=True`.** Video works, every other endpoint hangs.

**Encoding JPEG inside the stream generator.** Then encoding cost multiplies by
the number of viewers. Encode once in the pipeline; every viewer shares the
bytes.

**Blocking on `cv2.imshow` in a headless service.** It requires a display and
raises or hangs without one. `--show` exists for desktop debugging only.

**Assuming the browser closes the stream cleanly.** It does not always; the
generator may keep running after a tab closes. Harmless here because the loop
only reads a variable, but it would be a leak if it held resources.

---

## 9. Interview questions

**Q: How does the video stream work without a video codec?**
MJPEG over `multipart/x-mixed-replace`. The response never terminates; each part
is a complete JPEG with a boundary marker. The browser repaints on each part, so
a plain `<img>` tag shows live video with no JavaScript. Inefficient in
bandwidth, unbeatable in simplicity and latency.

**Q: How do you stop the web server from slowing inference?**
The pipeline publishes its most recent JPEG to a shared variable and never waits
for a consumer. The stream reads whatever is there, rate-limited to 15 FPS. A
slow or absent viewer causes dropped frames, never a stalled pipeline. A queue
would create back-pressure and let the browser dictate line speed.

**Q: Why `threaded=True`?**
`/stream` is an infinite response. Single-threaded, it monopolises the only
handler and every other route hangs.

**Q: Why is the reloader disabled?**
It runs the application twice. With a pipeline started at import, that is two
pipelines competing for one camera and one database.

**Q: Is this secure?**
No, and deliberately so. No authentication, no TLS, development server. It is a
demonstrator on a trusted LAN. Production would put it behind a reverse proxy
with auth and run it under a real WSGI server — configuration, not code.

**Q: Why show live and database statistics separately?**
They answer different questions. Live shows current health; database shows
cumulative history. Merging them would hide a restart, which is exactly the
event you most want to notice.

---

## 10. Summary

- Flask serves a **live MJPEG stream** plus JSON statistics from the same
  process that runs inspection.
- MJPEG is just **JPEGs concatenated in one endless HTTP response**. No codec,
  no player, no JavaScript.
- The pipeline **publishes** its latest frame; the stream **reads** it. No
  queue, so no back-pressure — a slow browser can never slow the line.
- `threaded=True` or every non-stream route hangs. `use_reloader=False` or you
  get two pipelines.
- Stream is capped at **15 FPS** so JPEG encoding does not compete with
  inference.
- Live and database statistics are shown separately, and the active config is
  included so a screenshot is self-describing.
- Not production-grade: no auth, development server. Honest scope, one config
  change away from better.

Next: [17 — GPIO, Arduino and the relay](17_gpio_arduino_relay.md).
