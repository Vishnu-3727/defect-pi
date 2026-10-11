#pragma once

// ── network ───────────────────────────────────────────────────────────────────
// ESP32 is 2.4 GHz ONLY. A 5 GHz-only hotspot will never connect.
#define WIFI_SSID   "CHANGE_ME"
#define WIFI_PASS   "CHANGE_ME"
#define PI_URL      "http://192.168.1.50:8000/ingest"
#define NODE_NAME   "motor1"

// ── sampling ──────────────────────────────────────────────────────────────────
#define FS_HZ        1000   // MPU-6050 output data rate (its hard ceiling)
#define WINDOW_N     1000   // samples per feature window -> 1 packet/second
#define I2C_HZ       400000
#define MPU_ADDR     0x68   // 0x69 if AD0 is pulled high

// ACCEL_CONFIG AFS_SEL. 0x00=+-2g 0x08=+-4g 0x10=+-8g 0x18=+-16g
// Start at +-8g, then check the reported clip count. Non-zero -> widen the range.
#define ACCEL_RANGE_REG   0x10
#define ACCEL_LSB_PER_G   4096.0f

// ── pins ──────────────────────────────────────────────────────────────────────
#define PIN_SDA   21
#define PIN_SCL   22
#define PIN_LED    2   // on-board LED: solid = healthy, blinking = alarm/offline

// ── motor (L298N) ─────────────────────────────────────────────────────────────
#define PIN_ENA   25   // PWM
#define PIN_IN1   26
#define PIN_IN2   27
#define MOTOR_DUTY 180 // 0-255
