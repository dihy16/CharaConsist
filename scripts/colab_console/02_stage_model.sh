#!/usr/bin/env bash
# Copy the configured Drive checkpoint to the configured local Colab path.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/manual_common.sh"
load_manual_config
require_colab
remote_exec stage_model.py
remote_exec check_model_copy.py 300
