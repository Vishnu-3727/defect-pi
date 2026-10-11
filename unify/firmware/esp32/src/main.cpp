// UNIFY-RUL sensor node — WEEK 1 SKELETON
//
//   core 1 : 1 kHz hardware-timer ISR -> sampler task -> I2C burst read -> ring
//            buffer -> every WINDOW_N samples, compute features -> queue
//   core 0 : network task -> JSON -> HTTP POST to the Pi
//
// The split matters: a blocking POST can take 200 ms, which would wreck the
// sample timing if it lived in the same task. See docs/09-communication.md
//
// FEATURE ORDER (must match edge/app.py :: FEATURE_NAMES):
//   ax_rms ax_kurt ax_crest  ay_rms ay_kurt ay_crest  az_rms az_kurt az_crest

#include <Arduino.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <Wire.h>
#include <stdarg.h>

#include "config.h"

static const int N_AXES = 3;
static const int N_FEAT = N_AXES * 3;

struct Packet {
  float feat[N_FEAT];
  uint32_t clip;
  uint32_t jitter_us;
};

static QueueHandle_t txQueue;
static TaskHandle_t samplerTask;
static hw_timer_t *timer = nullptr;

static float buf[N_AXES][WINDOW_N];

// ── MPU-6050 ──────────────────────────────────────────────────────────────────
static void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

static bool mpuInit() {
  Wire.begin(PIN_SDA, PIN_SCL, I2C_HZ);
  Wire.setTimeOut(50);  // motor EMI can hang the bus; never block forever

  Wire.beginTransmission(MPU_ADDR);
  if (Wire.endTransmission() != 0) return false;

  mpuWrite(0x6B, 0x00);            // PWR_MGMT_1: wake
  delay(50);
  mpuWrite(0x1A, 0x00);            // CONFIG: DLPF_CFG=0 -> 260 Hz accel bandwidth
  mpuWrite(0x19, 0x00);            // SMPLRT_DIV=0 -> 1 kHz
  mpuWrite(0x1C, ACCEL_RANGE_REG); // ACCEL_CONFIG
  return true;
}

// Reads ACCEL_XOUT_H..ACCEL_ZOUT_L as one burst so the three axes are
// time-coherent. Returns false on a bus fault.
static bool mpuRead(float *ax, float *ay, float *az, bool *clipped) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, 6, true) != 6) return false;

  int16_t raw[3];
  for (int i = 0; i < 3; i++) raw[i] = (int16_t)((Wire.read() << 8) | Wire.read());

  *clipped = false;
  for (int i = 0; i < 3; i++)
    if (raw[i] == 32767 || raw[i] == -32768) *clipped = true;

  *ax = raw[0] / ACCEL_LSB_PER_G;
  *ay = raw[1] / ACCEL_LSB_PER_G;
  *az = raw[2] / ACCEL_LSB_PER_G;
  return true;
}

// ── features ──────────────────────────────────────────────────────────────────
// Two-pass over the window: mean (the DC / gravity term), then central moments.
// Without the DC removal the ~1 g on the vertical axis dominates RMS and every
// feature looks constant forever. Keep this identical to unify/common/features.py.
static void axisFeatures(const float *x, int n, float *rms, float *kurt, float *crest) {
  double mean = 0.0;
  for (int i = 0; i < n; i++) mean += x[i];
  mean /= n;

  double m2 = 0.0, m4 = 0.0, peak = 0.0;
  for (int i = 0; i < n; i++) {
    double d = x[i] - mean;
    double d2 = d * d;
    m2 += d2;
    m4 += d2 * d2;
    if (fabs(d) > peak) peak = fabs(d);
  }
  m2 /= n;
  m4 /= n;

  *rms   = (float)sqrt(m2);
  *kurt  = m2 > 1e-12 ? (float)(m4 / (m2 * m2)) : 0.0f;  // ~3.0 for Gaussian
  *crest = *rms > 1e-9f ? (float)(peak / *rms) : 0.0f;
}

// ── sampling ──────────────────────────────────────────────────────────────────
void IRAM_ATTR onTimer() {
  BaseType_t woken = pdFALSE;
  // No I2C in an ISR. Just wake the sampler.
  vTaskNotifyGiveFromISR(samplerTask, &woken);
  if (woken) portYIELD_FROM_ISR();
}

static void samplerLoop(void *) {
  int idx = 0;
  uint32_t clip = 0;
  uint32_t lastUs = micros();
  uint32_t maxJitter = 0;

  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    uint32_t nowUs = micros();
    uint32_t dt = nowUs - lastUs;
    lastUs = nowUs;
    uint32_t expected = 1000000UL / FS_HZ;
    uint32_t jitter = dt > expected ? dt - expected : expected - dt;
    if (jitter > maxJitter) maxJitter = jitter;

    float ax, ay, az;
    bool clipped;
    if (!mpuRead(&ax, &ay, &az, &clipped)) {
      mpuInit();  // bus recovery — motor EMI is the usual cause
      continue;
    }
    if (clipped) clip++;

    buf[0][idx] = ax;
    buf[1][idx] = ay;
    buf[2][idx] = az;

    if (++idx == WINDOW_N) {
      Packet p;
      for (int a = 0; a < N_AXES; a++)
        axisFeatures(buf[a], WINDOW_N, &p.feat[a * 3], &p.feat[a * 3 + 1], &p.feat[a * 3 + 2]);
      p.clip = clip;
      p.jitter_us = maxJitter;

      xQueueSend(txQueue, &p, 0);  // never block the sampler
      idx = 0;
      clip = 0;
      maxJitter = 0;
    }
  }
}

// ── network ───────────────────────────────────────────────────────────────────
static void wifiEnsure() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) {
    delay(250);
    digitalWrite(PIN_LED, i & 1);
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("wifi ok, ip=%s rssi=%d\n", WiFi.localIP().toString().c_str(), WiFi.RSSI());
  }
}

static void networkLoop(void *) {
  static uint32_t seq = 0;
  WiFiClient client;
  HTTPClient http;
  Packet p;

  for (;;) {
    xQueueReceive(txQueue, &p, portMAX_DELAY);
    wifiEnsure();
    if (WiFi.status() != WL_CONNECTED) continue;  // ponytail: no store-and-forward yet, week 10

    // snprintf returns the length it WOULD have written, which can exceed the
    // buffer. Advancing by that value and then passing `sizeof(body) - n` gives a
    // negative length that converts to a huge size_t — a buffer overrun. Track the
    // remaining space explicitly and bail if we run out.
    char body[512];
    size_t n = 0;
    bool truncated = false;

    auto put = [&](const char *fmt, ...) {
      if (truncated || n >= sizeof(body) - 1) { truncated = true; return; }
      va_list ap;
      va_start(ap, fmt);
      int w = vsnprintf(body + n, sizeof(body) - n, fmt, ap);
      va_end(ap);
      if (w < 0 || (size_t)w >= sizeof(body) - n) { truncated = true; return; }
      n += (size_t)w;
    };

    put("{\"node\":\"%s\",\"seq\":%u,\"ts_ms\":%lu,\"clip\":%u,\"jitter_us\":%u,\"feat\":[",
        NODE_NAME, seq, (unsigned long)millis(), p.clip, p.jitter_us);
    for (int i = 0; i < N_FEAT; i++) put("%s%.6f", i ? "," : "", p.feat[i]);
    put("]}");

    if (truncated) {
      Serial.println("packet did not fit in the buffer — raise sizeof(body)");
      continue;
    }

    http.begin(client, PI_URL);
    http.setReuse(true);
    http.setTimeout(2000);
    http.addHeader("Content-Type", "application/json");
    int code = http.POST(body);
    if (code == 200) {
      seq++;
      digitalWrite(PIN_LED, HIGH);
    } else {
      Serial.printf("POST failed: %d\n", code);
      digitalWrite(PIN_LED, LOW);
    }
    http.end();
  }
}

// ── setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  pinMode(PIN_LED, OUTPUT);

  // motor: constant duty for now. Week 9 sweeps this to vary the operating point.
  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);
  digitalWrite(PIN_IN1, HIGH);
  digitalWrite(PIN_IN2, LOW);
  ledcSetup(0, 20000, 8);       // ch 0, 20 kHz (above audible), 8-bit
  ledcAttachPin(PIN_ENA, 0);
  ledcWrite(0, MOTOR_DUTY);

  while (!mpuInit()) {
    Serial.println("MPU-6050 not responding on 0x68 — check wiring / AD0");
    delay(1000);
  }
  Serial.println("MPU-6050 ok");

  txQueue = xQueueCreate(8, sizeof(Packet));

  xTaskCreatePinnedToCore(samplerLoop, "sampler", 4096, nullptr, 5, &samplerTask, 1);
  xTaskCreatePinnedToCore(networkLoop, "net", 8192, nullptr, 3, nullptr, 0);

  // arduino-esp32 2.x timer API. On 3.x this becomes:
  //   timer = timerBegin(FS_HZ); timerAttachInterrupt(timer, &onTimer);
  //   timerAlarm(timer, 1, true, 0);
  timer = timerBegin(0, 80, true);            // prescaler 80 -> 1 MHz tick
  timerAttachInterrupt(timer, &onTimer, true);
  timerAlarmWrite(timer, 1000000UL / FS_HZ, true);
  timerAlarmEnable(timer);
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(5000));
  Serial.printf("heap=%u rssi=%d queue=%u\n",
                ESP.getFreeHeap(), WiFi.RSSI(), uxQueueMessagesWaiting(txQueue));
}
