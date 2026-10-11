/*
 * Ejector controller for the defect inspection system.
 * Target: Arduino Nano (ATmega328P), 5 V.
 *
 * The Raspberry Pi sends one byte over USB serial; this sketch turns it into a
 * relay pulse of a guaranteed width.
 *
 *   'E'  eject   pulse the relay for PULSE_MS, then reply "OK"
 *   'P'  ping    reply "PONG", used to confirm the link is alive
 *   'S'  status  reply with the ejection count
 *
 * WHY THIS BOARD EXISTS AT ALL
 *
 * The Pi has GPIO pins and could drive the relay directly. Three reasons it
 * does not, in order of importance:
 *
 *  1. Timing. The Pi runs a general-purpose Linux kernel with no real-time
 *     guarantee. A 250 ms pulse written from Python can stretch by tens of
 *     milliseconds when the scheduler is busy -- and it is busy, because the
 *     same CPU is running the neural network. The ATmega has one job and an
 *     interrupt-free loop, so the pulse is the width the code says it is.
 *  2. Electrical isolation. A solenoid is an inductive load. When current
 *     stops, the collapsing magnetic field produces a reverse voltage spike
 *     that can be hundreds of volts. Two isolation barriers (the Arduino, then
 *     the relay opto-coupler) sit between that spike and a device with a
 *     filesystem to corrupt. A destroyed Nano costs a few hundred rupees; a
 *     destroyed Pi 5 costs the project.
 *  3. Voltage. Pi GPIO is 3.3 V and NOT 5 V tolerant. Many relay boards want a
 *     5 V logic high. The Nano is natively 5 V.
 *
 * WIRING
 *
 *   Nano D7 ---------> relay module IN
 *   Nano 5V ---------> relay module VCC        (see the warning below)
 *   Nano GND --------> relay module GND
 *   Nano USB --------> Raspberry Pi USB        (power and serial together)
 *
 *   Solenoid: driven from its OWN supply through the relay's COM/NO contacts.
 *
 *              +24V ---- [ relay COM ]
 *                          [ relay NO ] ---- solenoid +
 *                                            solenoid -  ---- 24V GND
 *
 *   A flyback diode (1N4007) MUST sit across the solenoid terminals, cathode
 *   (banded end) to the positive side. Without it the relay contacts arc and
 *   weld shut within a few thousand operations.
 *
 * WARNING -- do not power the solenoid from the Arduino or the Pi. A solenoid
 * draws amps. The Nano's regulator supplies milliamps. Use a separate supply
 * and tie only the grounds together.
 *
 * NOT YET TESTED ON HARDWARE. No board was available when this was written;
 * it compiles and the protocol matches src/inspector/actuator.py. Verify the
 * relay polarity (ACTIVE_LOW below) before connecting a solenoid.
 */

const uint8_t RELAY_PIN = 7;
const uint8_t LED_PIN   = LED_BUILTIN;

// Most cheap relay modules are active LOW: the input pin must be pulled to
// ground to energise the coil. Get this backwards and the relay sits closed
// from power-on, holding the solenoid energised until it overheats. Check your
// board and set this correctly BEFORE wiring the solenoid.
const bool ACTIVE_LOW = true;

// Pulse width. Long enough for the solenoid to travel and push the part clear,
// short enough not to overheat the coil. Tune on the rig; start at 250 ms.
const unsigned long PULSE_MS = 250;

// Ignore a second eject command arriving within this window. Protects the
// mechanism if the host ever repeats a command, and bounds the duty cycle.
const unsigned long MIN_INTERVAL_MS = 300;

unsigned long ejectCount = 0;
unsigned long lastEjectMs = 0;

void relayOff() { digitalWrite(RELAY_PIN, ACTIVE_LOW ? HIGH : LOW); }
void relayOn()  { digitalWrite(RELAY_PIN, ACTIVE_LOW ? LOW  : HIGH); }

void setup() {
  // Set the level BEFORE the pin becomes an output. Doing it the other way
  // round drives the pin LOW for a few microseconds, which on an active-low
  // board fires the solenoid every time the Arduino resets -- and it resets
  // every time the host opens the serial port.
  digitalWrite(RELAY_PIN, ACTIVE_LOW ? HIGH : LOW);
  pinMode(RELAY_PIN, OUTPUT);
  relayOff();

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.begin(9600);
  while (!Serial) { ; }
  Serial.println("READY");
}

void eject() {
  unsigned long now = millis();

  // millis() overflows after about 49 days. Subtracting unsigned values makes
  // the comparison overflow-safe; comparing the absolute values would not be.
  if (ejectCount > 0 && (now - lastEjectMs) < MIN_INTERVAL_MS) {
    Serial.println("BUSY");
    return;
  }

  digitalWrite(LED_PIN, HIGH);
  relayOn();
  delay(PULSE_MS);
  relayOff();
  digitalWrite(LED_PIN, LOW);

  lastEjectMs = now;
  ejectCount++;
  Serial.println("OK");
}

void loop() {
  if (!Serial.available()) {
    return;
  }

  char command = Serial.read();

  switch (command) {
    case 'E':
      eject();
      break;

    case 'P':
      Serial.println("PONG");
      break;

    case 'S':
      Serial.print("COUNT ");
      Serial.println(ejectCount);
      break;

    case '\n':
    case '\r':
      break;                      // line endings from a serial monitor

    default:
      Serial.print("ERR ");
      Serial.println(command);
      break;
  }
}
