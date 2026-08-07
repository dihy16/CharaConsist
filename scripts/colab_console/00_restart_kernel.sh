#!/usr/bin/env bash
# Restart the configured kernel before switching worker type or source version.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/manual_common.sh"
load_manual_config
require_colab
colab restart-kernel -s "$SESSION"
