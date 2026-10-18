#!/usr/bin/env bash
_omni_nccl_environment=$(LD_PRELOAD= python "$(dirname "${BASH_SOURCE[0]}")/omni_art_joint32/transport.py" shell-env) || return
eval "$_omni_nccl_environment"
unset _omni_nccl_environment
