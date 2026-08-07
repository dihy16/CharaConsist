#!/usr/bin/env bash
# Show current Colab status and recent worker output.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/manual_common.sh"
load_manual_config
require_colab
colab status -s "$SESSION"
colab log -s "$SESSION" -n "${1:-100}"
