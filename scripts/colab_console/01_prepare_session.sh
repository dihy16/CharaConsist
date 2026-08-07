#!/usr/bin/env bash
# Create/mount a Colab session and upload the shared remote configuration.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/manual_common.sh"
load_manual_config
require_colab

STATUS_OUTPUT="$(colab status -s "$SESSION" 2>&1 || true)"
if ! printf '%s\n' "$STATUS_OUTPUT" | grep -q 'Hardware:'; then
  colab new -s "$SESSION" --gpu "$GPU"
fi
colab drivemount -s "$SESSION"
colab upload -s "$SESSION" "$MANUAL_CONFIG" /content/characonsist_console_config.json
printf 'Session ready: %s\n' "$SESSION"
