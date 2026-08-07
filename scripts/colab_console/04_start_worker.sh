#!/usr/bin/env bash
# Start the persistent worker for lambda, component, role, binding, or routing conditions.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/manual_common.sh"
load_manual_config
require_colab

MODE="${1:-lambda}"
case "$MODE" in
  lambda) remote_exec start_worker.py ;;
  component) remote_exec start_component_worker.py ;;
  role) remote_exec start_role_action_worker.py ;;
  binding) remote_exec start_action_binding_worker.py ;;
  routing) remote_exec start_entity_routing_worker.py ;;
  *) fail "Worker mode must be lambda, component, role, binding, or routing." ;;
esac
