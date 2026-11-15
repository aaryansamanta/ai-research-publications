# Reorganization log

This repository was arranged from `cosmos.zip` (UCLA COSMOS 2026, Cluster 1). **No file was edited.**
Every file was either left at the same path inside the project, moved/renamed, or (four items) dropped.
Moved files are byte-identical to the originals.

- Left in place (relative to the old `InfernoTactics-main/`): **191** files
- Moved or renamed: **32** files
- Dropped: **4** items

## Moved / renamed

| Original | New location |
|---|---|
| `**Important** COSMOS Certificate.jpeg` | `_private/COSMOS_certificate.jpeg` |
| `COSMOS Final Project Thought Doc.docx` | `_private/COSMOS_project_thought_doc.docx` |
| `Cosmos-V4.docx` | `_private/COSMOS_statement_of_interest.docx` |
| `best_model/src/env/inferno_env.py` | `archive/best_model_v8_env/src/env/inferno_env.py` |
| `integration/backend.log` | `archive/integration_logs/backend.log` |
| `integration/server_live.err.log` | `archive/integration_logs/server_live.err.log` |
| `integration/server_live.out.log` | `archive/integration_logs/server_live.out.log` |
| `integration/server_smoke.err.log` | `archive/integration_logs/server_smoke.err.log` |
| `integration/server_smoke.out.log` | `archive/integration_logs/server_smoke.out.log` |
| `infernotactics/fire_spread_test.txt` | `archive/scratch/fire_spread_test.txt` |
| `integration/temp.js` | `archive/scratch/integration_temp.js` |
| `integration/test.js` | `archive/scratch/integration_test.js` |
| `test_cesium.js` | `archive/scratch/test_cesium.js` |
| `integration/Cesium.js` | `archive/unused/Cesium.js` |
| `Simulation3D.html` | `demo/Simulation3D.html` |
| `PROJECT_CONTEXT.md` | `docs/PROJECT_CONTEXT.md` |
| `infernotactics/README.md` | `docs/archive/README_infernotactics_legacy.md` |
| `README.md` | `docs/archive/README_v10_original.md` |
| `integration/cesium_after_fix.png` | `docs/figures/integration_debug/cesium_after_fix.png` |
| `integration/cesium_debug.png` | `docs/figures/integration_debug/cesium_debug.png` |
| `integration/cesium_final_debug.png` | `docs/figures/integration_debug/cesium_final_debug.png` |
| `infernotactics/TUTORIAL.md` | `docs/guides/TUTORIAL.md` |
| `COSMOS Final.pdf` | `docs/poster/InfernoCommand_poster.pdf` |
| `COSMOS Final.pptx` | `docs/poster/InfernoCommand_poster.pptx` |
| `Wildfire Command.docx` | `docs/writeup/Wildfire_Command_outline.docx` |
| `cache/1e84620de19678e4f166da292b3f69d01a105cf5.json` | `infernotactics/cache/1e84620de19678e4f166da292b3f69d01a105cf5.json` |
| `infernotactics/src/test_device.py` | `infernotactics/scripts/test_device.py` |
| `infernotactics/test_heuristic.py` | `infernotactics/scripts/test_heuristic.py` |
| `infernotactics/test_new_reward.py` | `infernotactics/scripts/test_new_reward.py` |
| `infernotactics/src/test_scatter.py` | `infernotactics/scripts/test_scatter.py` |
| `infernotactics/src/test_scatter2.py` | `infernotactics/scripts/test_scatter2.py` |
| `Response_delay` | `tools/camera_response_delay.py` |

## Dropped

| Item | Reason |
|---|---|
| `InfernoTactics-main.zip` | duplicate of InfernoTactics-main/ (verified identical) |
| `integration/__pycache__/server.cpython-313.pyc` | generated / OS junk |
| `integration/__pycache__/export_cesium_data.cpython-313.pyc` | generated / OS junk |
| `integration/__pycache__/app.cpython-311.pyc` | generated / OS junk |

(`.DS_Store` and `__MACOSX/` resource-fork files from the macOS zip were also not carried over.)

## Why things went where they did

- **`integration/` and `infernotactics/` stay siblings.** `integration/app.py` and `server.py` find the model
  code and checkpoints via `../infernotactics/...`, and `data_pipeline/config.py` finds `data/` relative to itself.
  Keeping that relationship means no code needed editing.
- **`_private/` is git-ignored.** It holds the COSMOS statement of interest (personal application essays),
  the team's scratch thought doc (contains unrelated personal notes and an off-topic remark about a named person),
  and the certificate. Move them to `docs/` deliberately if you want them public.
- **`archive/`** holds things that are superseded or scratch but that you may still want:
  the older env copy from `best_model/`, `temp.js`/`test.js` (draft copies of the Cesium page script),
  and old server logs. `archive/unused/Cesium.js` (13 MB) is not referenced by anything — `Simulation.html`
  loads Cesium 1.115 from the CDN — so it is git-ignored and safe to delete.
- **`Response_delay`** had no extension but is a Python script (ALERTCalifornia camera coverage / detection delay);
  it became `tools/camera_response_delay.py`. Nothing imports it (a comment in `integration/app.py` notes that its
  delay-mapping rules came from this script).
- **`cache/`** is the osmnx Overpass cache; it moved next to the scripts that write it (`infernotactics/cache/`).

## Things to know / fix (not changed here)

1. **Two different servers in `integration/`.** `app.py` (routes `/world`, `/simulate`) is the one `Simulation.html`
   talks to, and it serves the `.glb` truck/helicopter models from `integration/`. `server.py` (routes `/api/reset`,
   `/api/step`) drives the client in `integration/static/app.js`, but the HTML page that client needed was overwritten:
   `server.py` now returns `Simulation.html`, which never loads `static/app.js`. So `integration/README.md`'s
   `uvicorn server:app` recipe will show a page that can't reach its API. The root README documents the `app.py` route.
2. **Missing from the upload** but referenced by `docs/PROJECT_CONTEXT.md`: `src/viz/` (`export_trajectory.py`,
   `build_player.py`, `render_basemap.py` — the scripts that regenerate `demo/Simulation3D.html`),
   `best_model/inferno_best_model.pt` and its grid/roads, `context.txt`, and `integration-instructions.txt`.
   `demo/Simulation3D.html` itself is present and self-contained.
3. **`requirements.txt` is incomplete.** The code also imports `rich`, `folium` (tools only), and uses TensorBoard;
   `torch-directml` is optional (Windows/AMD).
4. **Hard-coded local paths.** The scripts in `infernotactics/scripts/` start with
   `sys.path.insert(0, r'A:\AI\InfernoTactics\infernotactics\src')`, so they only run on the original machine until that line is
   changed. `docs/guides/TUTORIAL.md` and `docs/archive/README_v10_original.md` use the same paths.
   The archived logs also contain a personal Google Drive path — keep `archive/` out of any public repo, or delete the logs.
5. **Stale docs, archived not deleted.** `docs/archive/README_infernotactics_legacy.md` describes `train_zonehead_randign.py`
   and Python 3.14 setup for code that no longer exists; `docs/archive/README_v10_original.md` still describes the current
   v10 pipeline correctly but points at missing files. The root `README.md` replaces both.
6. **Project name.** Code says *InfernoTactics*, the poster says *InfernoCommand*, the write-up says *Wildfire Command*.
   The repo uses the code's name. Rename the folder if you prefer the poster's.
