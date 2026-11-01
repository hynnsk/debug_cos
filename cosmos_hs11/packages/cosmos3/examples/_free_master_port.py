# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""cosmos_hs11: print the first free TCP port >= START (default span 200), probing like torchrun's TCPStore does
(SO_REUSEADDR, so a port that is merely in TIME_WAIT from a finished job still counts as free; one that another
torchrun / server is LISTENing on does not). Used by _sft_launcher_common.sh to bump MASTER_PORT when the recipe's
hard-coded port is already taken on the node. Usage: python examples/_free_master_port.py START [SPAN]"""

import socket
import sys


def first_free_port(start: int, span: int = 200) -> int | None:
    for port in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("", port))
            except OSError:
                continue
            return port
    return None


if __name__ == "__main__":
    start = int(sys.argv[1])
    span = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    port = first_free_port(start, span)
    if port is None:
        print(f"no free port in {start}..{start + span - 1}", file=sys.stderr)
        sys.exit(1)
    print(port)
