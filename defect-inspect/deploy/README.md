# Deploying to the Raspberry Pi 5

## Why a *user* service and not a system service

The unit installs under `~/.config/systemd/user/`, not `/etc/systemd/system/`.
Three reasons:

1. **It never runs as root.** The inspection loop opens a camera, a GPIO line,
   a SQLite file and a TCP socket. None of that needs root on a Pi, so nothing
   here should have it.
2. **`%h` resolves correctly.** In a system unit `%h` expands to `/root` even
   when `User=` is set, so paths have to be hardcoded to one machine. In a user
   unit it is that user's home, and the same file works for any username.
3. **GPIO already works without privilege.** Raspberry Pi OS puts the default
   user in the `gpio` group and ships a udev rule making `/dev/gpiochip*`
   group-writable. Confirm with `groups | grep gpio` before assuming it.

The cost is one extra step: user services normally stop when the last session
logs out, which for a headless Pi means at boot. `enable-linger` fixes that.

## Install

```bash
mkdir -p ~/.config/systemd/user
cp ~/defect-inspect/deploy/defect-inspect.service ~/.config/systemd/user/
sudo loginctl enable-linger "$USER"      # start at boot, not at login
systemctl --user daemon-reload
systemctl --user enable --now defect-inspect
```

## Check it

```bash
systemctl --user status defect-inspect
journalctl --user -u defect-inspect -f
curl -s localhost:5000/api/stats
```

The dashboard is then on `http://<pi-address>:5000` from any machine on the
network.

## Operating it

```bash
systemctl --user restart defect-inspect   # after editing configs/default.yaml
systemctl --user stop defect-inspect      # before running bench.py, or the
                                          # two compete for the same 4 cores
systemctl --user disable --now defect-inspect
```

**Stop the service before benchmarking.** `bench.py` pins four threads and so
does the pipeline; leaving both running produces numbers that measure the
contention, not the model.

## Power supply

This Pi runs from a **Nu Republic Cybotron 100 W GaN** charger, not the official
Raspberry Pi 27 W brick. That is a deliberate choice and it works, but the
constraint has to be understood rather than discovered mid-demo.

The Pi 5 negotiates **5 V only** — it never requests 9 V or 20 V. So of a
multi-port GaN charger's whole PD ladder, the single usable rung is the 5 V one,
and almost every such charger caps 5 V at 3 A. The official supply is unusual
precisely because it is a 5 V/5 A source. Measured here:

```
max_current              = 3000          (mA, i.e. 15 W ceiling)
usb_max_current_enable   = 0             (USB peripherals clamped to 600 mA)
usbpd_power_data_objects = 5 V / 3 A     (one profile, no PD ladder)
EXT5V idle               = 4.835 V
EXT5V under 4-core load  = 4.607 V       (228 mV sag)
```

4.607 V is just under the ~4.63 V warning threshold, which is why `dmesg` shows
`Undervoltage detected!` at boot. **It warns; it does not throttle.**
`vcgencmd get_throttled` gained no new bits across either the benchmark or a
deliberate four-core stress test, so the 36.7 FPS figure is valid.

The 228 mV is resistance, not capacity: the board draws around 1.2 A while 3 A
is available, so roughly 190 mΩ is sitting in the cable and connectors.

Consequences to design around, none of which need a new purchase:

- **Use the CSI Camera Module 3, not a USB webcam.** The firmware clamped the
  total USB budget to 600 mA when it saw a 3 A supply.
- **Power the LED ring light from its own supply**, never from the Pi.
- **The solenoid always needs a separate rail** regardless of the supply.
- Keep the other three charger ports empty; multi-port GaN units derate.

Check the state of any Pi with:

```bash
od -An -tu4 --endian=big /sys/firmware/devicetree/base/chosen/power/max_current
vcgencmd pmic_read_adc EXT5V_V
vcgencmd get_throttled
```

## Fan curve

`/boot/firmware/config.txt` brings the first fan step in 5 C earlier than stock:

```
dtparam=fan_temp0=45000,fan_temp0_hyst=5000,fan_temp0_speed=75
dtparam=fan_temp1=55000,fan_temp1_hyst=5000,fan_temp1_speed=125
dtparam=fan_temp2=65000,fan_temp2_hyst=5000,fan_temp2_speed=175
dtparam=fan_temp3=75000,fan_temp3_hyst=5000,fan_temp3_speed=250
```

A far more aggressive curve (full speed by 60 C) was tried first and reverted.
Two measurements killed it. Sustained four-thread inference tops out at
**57.6 C** against an 80 C soft-throttle point, so the extra cooling bought
margin that already existed; and the fan runs off the same 5 V rail as
everything else, costing **40 mV** at full speed on a supply already sagging
228 mV under CPU load. Real headroom, zero benefit, measurable cost.

Verify a benchmark was thermally valid with:

```bash
vcgencmd measure_temp; vcgencmd get_throttled
```

`get_throttled` must not gain new bits across the run. Bits 0-3 mean throttling
is happening now; bits 16-19 mean it happened at some point since boot.
