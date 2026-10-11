#!/usr/bin/env bash
# Run the UNIFY-RUL dashboard, on the Pi if it answers, else on this laptop.
#   ./run-dashboard.sh [host|local]   ($PI_HOST works the same as the argument)
# Pi user is vishnu. TCP 22 within 3 s decides whether the Pi is there.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAST_IP_FILE="$SCRIPT_DIR/.pi_last_ip"
PI_USER="vishnu"

tcp22_ok() {
  # $1 = host; return 0 if TCP 22 answers within ~3 s.
  if command -v python3 >/dev/null 2>&1; then
    python3 -c "import socket,sys; s=socket.socket(); s.settimeout(3); s.connect((sys.argv[1],22)); s.close()" "$1" >/dev/null 2>&1
    return $?
  fi
  if command -v nc >/dev/null 2>&1; then
    nc -z -w 3 "$1" 22 >/dev/null 2>&1
    return $?
  fi
  return 1
}

health_ok() {
  # $1 = full /health URL; return 0 only if UNIFY answers (JSON has "features").
  body=$(curl -s --max-time 3 "$1" 2>/dev/null)
  if [ -z "$body" ]; then
    return 1
  fi
  echo "$body" | grep -q '"features"'
}

save_pi_ip() {
  # $1 = host that answered; resolve to IPv4 and remember it. Skip on failure.
  h="$1"
  ip=""
  os=$(uname)
  if [ "$os" = "Darwin" ]; then
    out=$(ping -c1 -t1 "$h" 2>/dev/null)
    ip=$(echo "$out" | sed -n 's/.*(\([0-9][0-9.]*\)).*/\1/p' | head -n 1)
  else
    if command -v getent >/dev/null 2>&1; then
      ip=$(getent hosts "$h" 2>/dev/null | awk '{print $1}' | head -n 1)
    fi
  fi
  if [ -n "$ip" ]; then
    echo "$ip" > "$LAST_IP_FILE"
  fi
}

open_url() {
  os=$(uname)
  if [ "$os" = "Darwin" ]; then
    open "$1"
  else
    if command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$1" >/dev/null 2>&1 &
    else
      echo "Open $1 in your browser."
    fi
  fi
}

laptop_mode() {
  if ! command -v uv >/dev/null 2>&1; then
    echo 'uv not found. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh'
    exit 1
  fi
  PORT=""
  if [ -n "${UNIFY_PORT:-}" ]; then
    PORT="$UNIFY_PORT"
  else
    for p in 8000 8010 8020 8030; do
      if ! nc -z 127.0.0.1 $p >/dev/null 2>&1; then
        PORT=$p
        break
      fi
    done
  fi
  if [ -z "$PORT" ]; then
    echo "No free port among 8000, 8010, 8020, 8030."
    exit 1
  fi
  echo "Using port $PORT."
  echo "Starting dashboard on this laptop..."
  cd "$SCRIPT_DIR/edge" || exit 1
  uv run --with-requirements requirements-pi.txt python -m uvicorn app:app --host 127.0.0.1 --port $PORT &
  SERVER_PID=$!
  i=0
  while [ $i -lt 90 ]; do
    sleep 1
    i=$((i + 1))
    if ! kill -0 $SERVER_PID 2>/dev/null; then
      echo "UNIFY server exited early"
      exit 1
    fi
    if health_ok "http://127.0.0.1:$PORT/health"; then
      break
    fi
  done
  if ! health_ok "http://127.0.0.1:$PORT/health"; then
    echo "Server did not answer /health within 90 s."
    kill $SERVER_PID 2>/dev/null
    exit 1
  fi
  uv run --with-requirements requirements-pi.txt python fake_node.py http://127.0.0.1:$PORT/ingest --degrade --loop &
  SIM_PID=$!
  trap 'kill $SERVER_PID $SIM_PID 2>/dev/null' EXIT
  open_url "http://127.0.0.1:$PORT/"
  echo "Press Enter (Windows) / Ctrl+C (mac) to stop"
  wait
}

ARG1=""
if [ $# -ge 1 ]; then
  ARG1="$1"
fi
PI_ENV=""
if [ -n "${PI_HOST:-}" ]; then
  PI_ENV="$PI_HOST"
fi

FORCE_LOCAL=0
ONLY_CANDIDATE=""
if [ "$ARG1" = "local" ]; then
  FORCE_LOCAL=1
else
  if [ -n "$ARG1" ]; then
    ONLY_CANDIDATE="$ARG1"
  else
    if [ -n "$PI_ENV" ]; then
      ONLY_CANDIDATE="$PI_ENV"
    fi
  fi
fi

PI_FOUND=""
if [ "$FORCE_LOCAL" = "0" ]; then
  if [ -n "$ONLY_CANDIDATE" ]; then
    if tcp22_ok "$ONLY_CANDIDATE"; then
      PI_FOUND="$ONLY_CANDIDATE"
    fi
  else
    if tcp22_ok "defect-pi.local"; then
      PI_FOUND="defect-pi.local"
    else
      if [ -f "$LAST_IP_FILE" ]; then
        saved=$(cat "$LAST_IP_FILE" | tr -d ' \t\r\n')
        if [ -n "$saved" ]; then
          if tcp22_ok "$saved"; then
            PI_FOUND="$saved"
          fi
        fi
      fi
    fi
  fi
fi

if [ -n "$PI_FOUND" ]; then
  save_pi_ip "$PI_FOUND"
  HEALTH_URL="http://$PI_FOUND:8000/health"
  PAGE_URL="http://$PI_FOUND:8000/"
  if health_ok "$HEALTH_URL"; then
    echo "Pi already serving at $PAGE_URL"
    open_url "$PAGE_URL"
    echo "To stop: ssh $PI_USER@$PI_FOUND demo stop"
    exit 0
  fi
  echo "Starting demo on the Pi..."
  ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$PI_USER@$PI_FOUND" demo start unify-rul
  if [ $? -ne 0 ]; then
    echo "ssh failed, falling back to laptop mode."
    PI_FOUND=""
  else
    i=0
    while [ $i -lt 30 ]; do
      sleep 1
      i=$((i + 1))
      if health_ok "$HEALTH_URL"; then
        break
      fi
    done
    if health_ok "$HEALTH_URL"; then
      open_url "$PAGE_URL"
      echo "To stop: ssh $PI_USER@$PI_FOUND demo stop"
      exit 0
    else
      echo "Pi did not serve /health within 30 s, falling back to laptop mode."
      PI_FOUND=""
    fi
  fi
fi

if [ -z "$PI_FOUND" ]; then
  laptop_mode
fi
