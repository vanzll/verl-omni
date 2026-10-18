#!/usr/bin/env bash
: "${SOURCE_CHECKOUT_DIR:?explicit new source root required}"
: "${SOURCE_FILES_MANIFEST:?explicit pinned file manifest required}"
: "${SOURCE_FILES_MANIFEST_SHA256:?}"
: "${SOURCE_SNAPSHOT_SHA256:?archive identity required}"
timeout --foreground 120 "${OMNI_CAMPAIGN_ROOT:?}/env/bin/python" \
  "${OMNI_CONTROL_DIR:?}/control.py" verify-source
export PYTHONPATH="$SOURCE_CHECKOUT_DIR"
export PYTHONDONTWRITEBYTECODE=1
