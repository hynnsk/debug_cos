# openpi_hs11

openpi fork with the cosmos_hs11 pipeline on π0.5: **Reptile meta pre-training between `pi05_base` and LIBERO fine-tuning**,
LeRobot v3.0 datasets (NVIDIA LIBERO_LeRobot_v3 + the five robot_fewshot embodiments), 24 LIBERO recipes
(`pi05_libero_v3_<suite>_<all|3ep|1ep>_<base|reptile>`), launchers and a closed-loop evaluation client.

Experiments are TOML files under `scripts/hs11/toml/` with same-named launchers under `scripts/hs11/`
(`launch_reptile_pi05_v1.sh`, `launch_<suite>_<all|3ep|1ep>_<base|reptile>_v1.sh`, `eval_libero_v3.sh`).
Read `docs/hs11_reptile_pi05.md` for setup, the data conventions and the exact commands.
