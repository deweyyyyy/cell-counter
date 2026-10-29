#!/bin/bash
# Run the Cell Counter server in the background on macOS:
# starts at login, restarts itself if it stops, keeps the Mac awake while serving,
# and writes everything it prints to server.log in this folder.
#
#   ./install-service.sh            install (or reinstall) and start
#   ./install-service.sh --remove   stop and remove
set -e
cd "$(dirname "$0")"
DIR="$(pwd)"
LABEL="app.cellcounter.server"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"

if [ "$1" = "--remove" ]; then
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Removed. The server no longer starts on its own."
  exit 0
fi

if [ "$(uname -s)" != "Darwin" ]; then echo "This installer is for macOS."; exit 1; fi
chmod +x start.sh

# a copy started by hand in Terminal would hold port 8000
if lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 && ! launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
  echo "Something is already using port 8000 (probably ./start.sh in another Terminal window)."
  echo "Stop it with Ctrl+C in that window, then run this again."
  exit 1
fi

# first run: build the private Python environment now, so the service starts quickly
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q numpy pillow
fi
if [ "$(uname -m)" = "arm64" ] && ! .venv/bin/python -c "import mlx.core" 2>/dev/null; then
  .venv/bin/pip install -q mlx || true
fi

mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/caffeinate</string><string>-i</string>
    <string>$DIR/.venv/bin/python</string><string>-u</string><string>$DIR/server.py</string>
  </array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>$DIR/server.log</string>
  <key>StandardErrorPath</key><string>$DIR/server.log</string>
</dict>
</plist>
EOF

launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$PLIST"
launchctl kickstart -k "$DOMAIN/$LABEL"

printf "Starting"
for i in $(seq 1 60); do
  if curl -s -m 2 -o /dev/null http://127.0.0.1:8000/api/info; then
    echo " done."
    echo "The server runs in the background now: it starts at login and restarts itself if it stops."
    echo "Log: $DIR/server.log    Stop it: ./install-service.sh --remove"
    exit 0
  fi
  printf "."; sleep 1
done
echo
echo "It didn't answer within a minute. The last lines of the log:"
tail -20 "$DIR/server.log"
exit 1
