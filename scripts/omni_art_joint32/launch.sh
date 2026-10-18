#!/usr/bin/env bash
set -eo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
set -u
control=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
case "${1:-}" in
  prepare)
    shift
    exec timeout --foreground 300 /usr/bin/python "$control/control.py" prepare "$@"
    ;;
  start)
    shift
    exec timeout --foreground 60 /usr/bin/python "$control/control.py" start "$@"
    ;;
  supervise)
    shift
    exec /usr/bin/python -u "$control/control.py" supervise "$@"
    ;;
  *) echo 'usage: bash launch.sh {prepare [options]|start MANIFEST|supervise MANIFEST}' >&2; exit 2 ;;
esac
