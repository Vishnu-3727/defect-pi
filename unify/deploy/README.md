# Running both demos on one Pi 5

UNIFY-RUL and `defect-inspect` share one Raspberry Pi 5. They are never shown at
the same time — one demo runs, gets closed, the next one opens. `Conflicts=` in
the two unit files makes that exclusion automatic: starting either one stops the
other. You cannot accidentally leave both running.

| | defect-inspect | UNIFY-RUL |
|---|---|---|
| Unit | `defect-inspect.service` | `unify-rul.service` |
| Port | 5000 | 8000 |
| Directory | `~/defect-inspect` | `~/unify` |

## One-time setup on the Pi

The Pi runtime is numpy + onnxruntime + FastAPI only. It must never import
torch — `tests/test_pipeline.py` in the sibling repo enforces the same rule
there, and `edge/inference.py` here depends on it staying true.

```bash
uv venv ~/unify/.venv
uv pip install --python ~/unify/.venv -r ~/unify/edge/requirements-pi.txt

mkdir -p ~/.config/systemd/user
cp ~/unify/deploy/unify-rul.service ~/.config/systemd/user/
cp ~/defect-inspect/deploy/defect-inspect.service ~/.config/systemd/user/
systemctl --user daemon-reload

cp ~/unify/deploy/unify-rul.desktop ~/Desktop/
cp ~/defect-inspect/deploy/defect-inspect.desktop ~/Desktop/
chmod +x ~/Desktop/*.desktop      # or right-click -> Allow Launching
```

Linger is already enabled on this Pi. If it ever is not:
`loginctl enable-linger $USER`.

## Switching demos

Double-click the desktop icon. That is the whole operation — the icon starts one
unit, systemd stops the other.

Keep two browser tabs open, `localhost:5000` and `localhost:8000`, and reload
whichever one you just switched to. The tabs survive the switch; only the
service behind them changes.

Equivalent from a terminal:

```bash
systemctl --user start unify-rul          # stops defect-inspect
systemctl --user start defect-inspect     # stops unify-rul
systemctl --user status unify-rul
journalctl --user -u unify-rul -f
```

## Simulated sensor feed

No ESP32 exists yet, so `unify-rul-feed.service` posts synthetic packets from
`edge/fake_node.py --degrade --loop` (healthy → failing → healthy again).
`Wants=` in `unify-rul.service` starts the feed with the server, and
`BindsTo=` in the feed unit stops it when the server stops. Copy it alongside
the server unit (`cp ~/unify/deploy/unify-rul-feed.service
~/.config/systemd/user/`) — the dashboard labels this feed as simulated.

## Which one comes up at boot

Only enable one, or the two `Conflicts=` declarations race at startup and the
winner is arbitrary:

```bash
systemctl --user enable defect-inspect    # the boot default
systemctl --user disable unify-rul        # started from its icon
```

## Notes

- Threads: leave `defect-inspect` at `num_threads: 4` and UNIFY-RUL at 2. Each
  demo owns the whole Pi while it runs, so there is nothing to split.
- Power: the Pi logs `Undervoltage detected!` under load on the 15 W GaN supply.
  It warns and has never throttled — `get_throttled` gained no new bits across
  any benchmark run. Expect the message; it is not a fault.
- The `Makefile` `deploy` target restarts `unify-rul` over SSH. It targets
  `vishnu@192.168.1.6` (eth0); override with `make deploy PI=vishnu@192.168.1.7`
  for wlan0.
