# libero_fewshot_gifs

Exports the few-shot LIBERO demonstrations (`<suite>_3ep_per_task_seed42.json`, 3 episodes x 10 tasks) of a suite as GIFs.
The outputs live next to this folder, one folder per task named by its text instruction, one GIF per episode:

    utils/libero_10_3ep_seed42_gifs/<text instruction>/ep{episode_index:03d}.gif
    utils/libero_goal_3ep_seed42_gifs/...        512x256: 3rd-person camera (left) | wrist camera (right), 20 fps, every frame
    utils/libero_object_3ep_seed42_gifs/...      + manifest.json (episode -> frames, fps, size, gif path)
    utils/libero_spatial_3ep_seed42_gifs/...
    utils/libero_goal_3ep_v2_gifs/      episodes ADDED in libero_goal_3ep_per_task_v2.json (hs11) + 10 extra browsing candidates
                                        for task 9 (extra_candidates_task9.json; not in any subset)
    utils/libero_spatial_3ep_v2_gifs/   episodes ADDED in libero_spatial_3ep_per_task_v2.json and _v3.json (hs11)

Regenerate / variants (CPU only, ~15 s per suite with 8 workers; existing GIFs are skipped unless --overwrite):

    bash utils/libero_fewshot_gifs/run.sh --suite libero_goal
    for s in libero_10 libero_goal libero_object libero_spatial; do bash utils/libero_fewshot_gifs/run.sh --suite $s; done
    bash utils/libero_fewshot_gifs/run.sh --suite libero_object --views image                      # 3rd-person only
    bash utils/libero_fewshot_gifs/run.sh --suite libero_10 --stride 2 --scale 0.5 --out-dir <dir> # 10 fps, 128 px/view (~1/8 size)
    bash utils/libero_fewshot_gifs/run.sh --suite libero_10 --episode-subset libero_10_10ep_per_task_seed42.json --out-dir <dir>
    bash utils/libero_fewshot_gifs/run.sh --suite libero_goal --episode-subset libero_goal_3ep_per_task_v2.json \
         --out-dir utils/libero_goal_3ep_v2_gifs --only-episodes 26,111                        # subset of episodes only
    bash utils/libero_fewshot_gifs/run.sh --suite libero_goal --out-dir <dir> --episodes 31,58,84      # arbitrary dataset episodes
         # (not necessarily subset members; task folder from the dataset's task text; listed under manifest "extra_episodes")

`--suite` sets the dataset root (`$COSMOS_STORAGE/data/LIBERO_LeRobot_v3/<suite>`), the bundled subset JSON (newest cosmos_hs*
repo) and the output folder; the JSON's `source_root_basename` and the task texts are checked against the dataset (`--force` skips).
Decoding: PyAV (libdav1d) over the LeRobot v3 AV1 mp4s, exact to the frame (counts match `length` in meta/episodes).
Encoding: ffmpeg palettegen/paletteuse (one palette per GIF); Pillow fallback with `--no-ffmpeg`.
