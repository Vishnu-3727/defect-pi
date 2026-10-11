#!/bin/bash
# Open the defect-inspect dashboard: Pi if reachable, else laptop.
PI_USER="vishnu"
PI_ENV="${PI_HOST:-}"
PI_HOST="${PI_HOST:-defect-pi.local}"
PORT="5000"
FORCE_LOCAL="0"
EXPLICIT="0"
if [ -n "$1" ]; then
  if [ "$1" = "local" ]; then FORCE_LOCAL="1"; else PI_HOST="$1"; EXPLICIT="1"; fi
elif [ -n "$PI_ENV" ]; then
  EXPLICIT="1"
fi
SCRIPT_DIR="$(dirname "$0")"
LAST_IP_FILE="$SCRIPT_DIR/.pi_last_ip"
http_ok() {
  curl -sf -o /dev/null --max-time 5 "$1" 2>/dev/null
}
open_browser() {
  if [ "$(uname)" = "Darwin" ]; then open "$1"; else xdg-open "$1"; fi
}
pi_reachable() {
  if [ "$(uname)" = "Darwin" ]; then
    nc -z -G 3 "$1" 22 >/dev/null 2>&1
  else
    timeout 3 bash -c "</dev/tcp/$1/22" >/dev/null 2>&1
  fi
}
wait_http() {
  i=0
  while [ "$i" -lt "$1" ]; do
    if http_ok "$2"; then return 0; fi
    sleep 1
    i=$((i + 1))
  done
  return 1
}
laptop_mode() {
  if ! command -v uv >/dev/null 2>&1; then
    echo 'uv not found. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh'
    exit 1
  fi
  cd "$(dirname "$0")" || exit 1
  ( if wait_http 60 "http://localhost:$PORT/"; then open_browser "http://localhost:$PORT/"; fi ) &
  uv run python dashboard/app.py --source bench_fixtures/images --loop
}
FOUND=""
TRIED=""
if [ "$FORCE_LOCAL" = "0" ]; then
  if [ "$EXPLICIT" = "1" ]; then
    TRIED="$PI_HOST"
    if pi_reachable "$PI_HOST"; then FOUND="$PI_HOST"; fi
  else
    TRIED="defect-pi.local"
    if pi_reachable "defect-pi.local"; then
      FOUND="defect-pi.local"
    else
      if [ -f "$LAST_IP_FILE" ]; then
        SAVED="$(head -n 1 "$LAST_IP_FILE" 2>/dev/null | tr -d ' \t\r\n')"
        if [ -n "$SAVED" ] && [ "$SAVED" != "defect-pi.local" ]; then
          TRIED="defect-pi.local, $SAVED"
          if pi_reachable "$SAVED"; then FOUND="$SAVED"; fi
        fi
      fi
    fi
  fi
  if [ -n "$FOUND" ]; then
    PI_HOST="$FOUND"
    RESOLVED=""
    if echo "$FOUND" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$'; then
      RESOLVED="$FOUND"
    elif [ "$(uname)" = "Darwin" ]; then
      RESOLVED="$(ping -c1 -t1 "$FOUND" 2>/dev/null | sed -n 's/.*(\([0-9][0-9.]*\)).*/\1/p' | head -n 1)"
    else
      RESOLVED="$(getent hosts "$FOUND" 2>/dev/null | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)"
    fi
    if [ -n "$RESOLVED" ]; then
      echo "$RESOLVED" > "$LAST_IP_FILE" 2>/dev/null || true
    fi
  fi
fi
if [ "$FORCE_LOCAL" = "0" ] && [ -n "$FOUND" ]; then
  PI_URL="http://$PI_HOST:$PORT/"
  if http_ok "$PI_URL"; then
    echo "Dashboard already running on Pi: $PI_URL"
    open_browser "$PI_URL"
    exit 0
  fi
  ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$PI_USER@$PI_HOST" demo start defect-inspect
  if [ "$?" -ne 0 ]; then
    echo "ssh to $PI_HOST failed. Falling back to laptop mode."
    laptop_mode
    exit $?
  fi
  if wait_http 30 "$PI_URL"; then open_browser "$PI_URL";
  else echo "Pi dashboard did not answer at $PI_URL within 30 s."; fi
  echo "To stop the Pi demo: ssh $PI_USER@$PI_HOST demo stop"
  exit 0
fi
if [ "$FORCE_LOCAL" = "0" ] && [ -z "$FOUND" ]; then echo "Pi not found (tried: $TRIED). Using laptop mode."; fi
laptop_mode
