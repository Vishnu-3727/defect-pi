"""The ejector, behind one interface with three backends.

  mock    logs the pulse; runs anywhere, needs no hardware
  serial  sends a command to an Arduino Nano over USB
  gpio    drives a Pi GPIO pin directly through lgpio

The pipeline calls `eject()` and never learns which one it got. That is the
whole point: the system was built and tested with no hardware at all, and
switching to a real solenoid is one line in configs/default.yaml.

Why an Arduino at all, when the Pi has GPIO pins: see docs/17. Short version --
the Pi's pins are 3.3 V, not 5 V tolerant, and it runs a non-real-time Linux
kernel, so a pulse width controlled from Python can jitter by tens of
milliseconds under load. An Arduino guarantees the pulse width in hardware and
gives a second layer of isolation between a 24 V inductive solenoid and a
device with a filesystem to corrupt.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

log = logging.getLogger(__name__)

# One byte, so the Arduino can act on it without parsing. See
# firmware/ejector/ejector.ino.
EJECT_COMMAND = b"E"
PING_COMMAND = b"P"


class Actuator(Protocol):
    name: str

    def eject(self) -> bool:
        """Fire the ejector. Returns True if the command was delivered."""

    def close(self) -> None:
        ...


class MockActuator:
    """Logs instead of firing. The default, and what CI and the demo use."""

    name = "mock"

    def __init__(self, pulse_ms: int = 250):
        self.pulse_ms = pulse_ms
        self.fired = 0

    def eject(self) -> bool:
        self.fired += 1
        log.info("MOCK EJECT #%d (%d ms pulse)", self.fired, self.pulse_ms)
        return True

    def close(self) -> None:
        log.info("mock actuator closed after %d ejections", self.fired)


class SerialActuator:
    """Arduino Nano over USB serial.

    The Arduino owns the pulse timing. We send one byte and return; we do not
    hold the line high for 250 ms from Python, because Python cannot promise
    250 ms.
    """

    name = "serial"

    def __init__(self, port: str, baud: int = 9600, pulse_ms: int = 250,
                 timeout: float = 1.0):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is not installed") from exc

        self.pulse_ms = pulse_ms
        self.port = port
        try:
            self.link = serial.Serial(port, baud, timeout=timeout, write_timeout=timeout)
        except serial.SerialException as exc:
            raise RuntimeError(
                f"could not open {port}. Check the Arduino is connected, the port "
                f"name is right (COMx on Windows, /dev/ttyUSB0 or /dev/ttyACM0 on "
                f"Linux), and no serial monitor is holding it open."
            ) from exc

        # Opening the port toggles DTR, which resets an Arduino Nano. It spends
        # roughly two seconds in the bootloader; anything sent during that
        # window is swallowed. This sleep is not superstition.
        time.sleep(2.0)
        self.link.reset_input_buffer()
        self.fired = 0
        log.info("serial actuator ready on %s", port)

    def eject(self) -> bool:
        try:
            self.link.write(EJECT_COMMAND)
            self.link.flush()
            self.fired += 1
            return True
        except Exception as exc:
            # A failed ejection must never stop the inspection loop. The part is
            # already logged as a reject in the database; losing the line is worse.
            log.error("serial eject failed: %s", exc)
            return False

    def close(self) -> None:
        try:
            self.link.close()
        except Exception:
            pass


class GpioActuator:
    """Direct Pi 5 GPIO through lgpio.

    Simpler than the Arduino path and one component cheaper, but the pulse width
    is at the mercy of the Linux scheduler, and the pin is directly exposed to
    whatever the relay board does. Use it when the relay module has its own
    opto-isolation and the timing tolerance is loose.

    RPi.GPIO does not work on a Pi 5: the GPIO controller moved into the RP1
    south bridge. lgpio and gpiozero are the supported paths.
    """

    name = "gpio"

    def __init__(self, pin: int, pulse_ms: int = 250):
        try:
            import lgpio
        except ImportError as exc:
            raise RuntimeError(
                "lgpio is not installed. On a Pi: uv sync --extra pi. "
                "Note RPi.GPIO does not support the Pi 5."
            ) from exc

        self._lgpio = lgpio
        self.pin = pin
        self.pulse_ms = pulse_ms
        self.chip = lgpio.gpiochip_open(4 if self._is_pi5() else 0)
        # Claim the pin already LOW. Claiming it without an initial value can
        # glitch the relay on startup, which fires the solenoid at boot.
        lgpio.gpio_claim_output(self.chip, pin, 0)
        self.fired = 0
        log.info("gpio actuator ready on BCM %d", pin)

    @staticmethod
    def _is_pi5() -> bool:
        try:
            with open("/proc/device-tree/model", "rb") as fh:
                return b"Raspberry Pi 5" in fh.read()
        except OSError:
            return False

    def eject(self) -> bool:
        try:
            self._lgpio.gpio_write(self.chip, self.pin, 1)
            time.sleep(self.pulse_ms / 1000.0)
            self._lgpio.gpio_write(self.chip, self.pin, 0)
            self.fired += 1
            return True
        except Exception as exc:
            log.error("gpio eject failed: %s", exc)
            return False

    def close(self) -> None:
        try:
            self._lgpio.gpio_write(self.chip, self.pin, 0)
            self._lgpio.gpiochip_close(self.chip)
        except Exception:
            pass


def build_actuator(cfg) -> Actuator:
    """Construct the configured backend, falling back to mock if it cannot open.

    Falling back rather than crashing is deliberate: an unplugged Arduino should
    degrade the system to inspection-and-logging, not stop it. The fallback is
    logged at WARNING so it can never pass unnoticed.
    """
    a = cfg.actuator
    try:
        if a.backend == "serial":
            return SerialActuator(a.port, a.baud, a.pulse_ms)
        if a.backend == "gpio":
            return GpioActuator(a.pin, a.pulse_ms)
    except RuntimeError as exc:
        log.warning("%s actuator unavailable (%s); falling back to mock", a.backend, exc)
        return MockActuator(a.pulse_ms)
    return MockActuator(a.pulse_ms)


def demo() -> None:
    """Self-check: the mock backend counts, and an unopenable port degrades."""
    from .config import load_config

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    mock = MockActuator(pulse_ms=120)
    assert mock.eject() is True
    assert mock.eject() is True
    assert mock.fired == 2
    mock.close()

    cfg = load_config()
    assert build_actuator(cfg).name == "mock", "default config should be mock"

    # A serial backend pointed at a port that cannot exist must degrade to mock,
    # not raise. Losing the ejector should never stop the inspection line.
    import dataclasses

    broken = dataclasses.replace(cfg.actuator, backend="serial", port="COM_NOPE")
    fallback = build_actuator(dataclasses.replace(cfg, actuator=broken))
    assert fallback.name == "mock", f"expected mock fallback, got {fallback.name}"

    print("actuator self-check passed: mock counts, bad serial port degrades to mock")


if __name__ == "__main__":
    demo()
