#!/usr/bin/env bash
# Fetch gs://openpi-assets/checkpoints/pi05_base/params into $OPENPI_DATA_HOME (anonymous GCS access, ~13 GB). CPU only.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
JAX_PLATFORMS=cpu "$PY" - <<'PYEOF'
from openpi.shared import download
p = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base/params", gs={"token": "anon"})
print("pi05_base params at", p)
PYEOF
