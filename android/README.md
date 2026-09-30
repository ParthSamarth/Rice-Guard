# RiceGuard AI — Android client + local API server

*Developed by: Parth Samarth*

This directory (`android/`) is the Android client. Its server counterpart lives at
`../server/` (FastAPI). Both sit **on top of** the already-validated, frozen Intelligence
Layer (`../pipeline/inference_pipeline.py` and everything it calls) — nothing here retrains,
fine-tunes, or modifies `models/yolo_yolov8n/best.pt` or `models/classifier_resnet50/best.pt`.
See `../PROJECT_README.md` for that layer's own documentation.

> **Build-verification disclosure.** This project was authored in an environment with no
> JDK, Android SDK, or Gradle installed (verified before writing any code — `java`, `gradle`
> were absent from `PATH`, and no `ANDROID_HOME` was set). Every Kotlin/Gradle file here was
> written to compile cleanly by careful manual review — package declarations, cross-file
> references, and every screen's call signature were systematically checked against their
> call sites — but **none of it has actually been run through a Kotlin/Gradle compiler**.
> The FastAPI server, by contrast, **was** installed, started, and exercised live end-to-end
> (see `../server/main.py`'s module docstring and the verification steps below) — it is a
> fully tested, working service, not just authored code. The first real build of the Android
> app must happen in Android Studio on a machine with the SDK installed; see
> **"⚠ FIRST REAL BUILD / DEVICE VERIFICATION"** below for exact steps.

```
                     Architecture
    ┌──────────────────────────┐
    │   CLIENT LAYER            │   Android (Kotlin, Jetpack Compose,
    │   android/                │   CameraX, Room, Retrofit)
    └─────────────┬────────────┘
                   │  HTTP POST multipart/form-data over local Wi-Fi
                   │  (phone and PC on the same network/router)
                   ▼
    ┌──────────────────────────┐
    │   APPLICATION LAYER       │   FastAPI + Uvicorn, one process,
    │   server/                 │   local-network-only (server/security.py)
    └─────────────┬────────────┘
                   │  calls RiceDiseasePipeline.run() directly —
                   │  no logic duplicated, no models reimplemented
                   ▼
    ┌──────────────────────────┐
    │   INTELLIGENCE LAYER      │   YOLOv8n @ 640 → ResNet50 → Grad-CAM →
    │   pipeline/ (unchanged)   │   Recommendation Engine (all pre-existing,
    │                            │   already validated — see PROJECT_README.md)
    └─────────────┬────────────┘
                   ▼
    ┌──────────────────────────┐
    │   DATA LAYER               │  recommendation/knowledge_base.json (PC)
    │                             │  server/tmp/<request_id>/ (temporary only)
    │                             │  Android: Room DB + private app storage
    └──────────────────────────┘
```

## PC setup (server)

1. This project's existing Python venv already has the ML dependencies (`torch`,
   `ultralytics`, etc. — see `../requirements.txt`). Add the server's own on top of it:
   ```
   ../.venv/Scripts/python.exe -m pip install -r requirements.txt
   ```
   (run from `server/`; installs `fastapi`, `uvicorn[standard]`, `python-multipart`)
2. Verify the final model checkpoints exist (the server refuses to start otherwise —
   see `PipelineService.verify_model_files()` in `server/pipeline_service.py`):
   ```
   models/yolo_yolov8n/best.pt
   models/classifier_resnet50/best.pt
   ```
3. Start the server:
   ```
   server/run_server.ps1
   ```
   or manually: `cd server; ../.venv/Scripts/python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000`
4. Find this PC's LAN IP (`ipconfig`, look for the IPv4 address on your Wi-Fi adapter —
   typically `192.168.x.x`). You'll enter this in the Android app's Settings screen.
5. Confirm it's up: open `http://<pc-ip>:8000/health` in a browser, or from the PC itself:
   ```
   curl http://localhost:8000/health
   ```
   Expect `{"status":"ok","service":"RiceGuard AI","models_ready":true,"yolo":true,"resnet50":true,...}`.
   API docs (Swagger UI) are auto-served at `http://<pc-ip>:8000/docs`.

**This was actually run and verified during development** (not just written): `/health` and
`/predict` were both exercised against real test images from every decision path (a
detection+agreement case, a no-detection/full-image case), plus error cases (corrupt file,
wrong MIME type, missing result file, path-traversal attempt) — all returned the expected
responses. See `server/main.py` for the full endpoint list.

## Android setup

1. Open the `android/` folder as a project in Android Studio.
2. Minimum SDK 26 (Android 8.0+), target/compile SDK 34.
3. Build and run on a **physical device** on the same Wi-Fi network as the PC (an emulator's
   virtual network usually can't reach the host PC's LAN IP directly — a physical phone is the
   supported path for this local-network design).
4. On first launch: Home → Settings → enter the PC's IP and port (default `8000`) →
   **Test Connection**. The app never guesses or hard-codes an IP (project brief section 19).

---

## ⚠ FIRST REAL BUILD / DEVICE VERIFICATION

**This code has never been compiled.** Every check possible without a compiler has been done —
see "Source-level verification performed" below — but the first actual Gradle build, on real
Kotlin/Compose/Room/CameraX toolchain, has not happened yet. That must be done on a machine with:

- **JDK 17**
- **Android SDK** (compileSdk/targetSdk 34, minSdk 26 already installed via the SDK Manager)
- **Android Studio** with an AGP version supporting Gradle 8.7 / AGP 8.5 (Koala/2024.1 or newer)

### Exact first steps, in order

1. Install/confirm the three prerequisites above (Android Studio's own installer handles JDK +
   SDK together; run **Tools → SDK Manager** and confirm Android 14 (API 34) is installed).
2. **File → Open...** → select the `android/` folder (not the repo root) → let Gradle sync.
   On first sync, Android Studio will generate the missing `gradle/wrapper/gradle-wrapper.jar`
   binary itself (this repo ships `gradle-wrapper.properties` pointing at Gradle 8.7, but not
   the jar, since a binary can't be authored as text) — this is expected, not an error.
3. Resolve any sync error using "Likely first-build issues" below before assuming the app logic
   itself is wrong.
4. **Run ▸ Run 'app'** with a physical Android device connected via USB (USB debugging enabled
   in Developer Options) or on the same Wi-Fi for wireless ADB — **not** the emulator (see
   "Android setup" step 3 above for why).
5. Start `server/run_server.ps1` on the PC first (see "PC setup" above — this half is already
   verified working), find the PC's LAN IP with `ipconfig`, then in the app: Home → Settings →
   enter that IP and port `8000` → **Test Connection** → confirm it shows "AI Server Connected."
6. Walk the acceptance flow once, end to end, on the physical device: Scan Rice Leaf → capture →
   confirm the preview appears → **Retake** once (confirm it returns to a live camera, not a
   frozen one) → capture again → **Use Photo** → confirm the staged-progress screen appears →
   confirm a Result screen appears with a real diagnosis → open Explanation and Recommendation →
   Done → confirm the scan now appears in History → open it → Delete Scan → confirm it's gone →
   Settings → toggle Save Scan History off → run one more scan → confirm History does **not**
   grow → Settings → Delete All Scan Data (if any remain) → confirm the count reaches 0.

### Likely first-build issues (and why they're not expected to be logic bugs)

- **A dependency version in `gradle/libs.versions.toml` has since been superseded.** Bump the
  specific `version.ref` line; the rest of the catalog doesn't need to change.
- **Compose compiler extension mismatch** — `composeOptions.kotlinCompilerExtensionVersion` in
  `app/build.gradle.kts` must match the Kotlin version for your installed AGP exactly; check the
  [Compose-Kotlin compatibility map](https://developer.android.com/jetpack/androidx/releases/compose-kotlin)
  if Studio flags this.
- **KSP/Room annotation-processing version drift** — `ksp` in the version catalog must track the
  exact Kotlin version (`<kotlin-version>-<ksp-build>`); if Studio suggests a different KSP
  version for your Kotlin version, take its suggestion.

None of the above are expected to require touching `.kt` application-logic files. If a build
error points *into* a screen/ViewModel/repository file rather than a Gradle config file, treat
that specifically as a real finding to investigate (see "Source-level verification performed").

### Source-level verification already performed (2026-08-19, before any compiler was available)

All of the following were checked directly against the source, not assumed:

1. Package declaration vs. directory path — all 51 Kotlin files, zero mismatches.
2. Every internal (`com.riceguard.ai.*`) import cross-checked against an actual declaration.
3. Every `AndroidManifest.xml` reference (`.RiceGuardApplication`, `.MainActivity`, the theme,
   `network_security_config.xml`, both launcher icon variants, `app_name`) confirmed to resolve.
4. Every `libs.*` accessor used in the Gradle files confirmed against `libs.versions.toml`'s
   actual `[libraries]`/`[plugins]` keys (31/31 resolve).
5. Room: `ScanRecord` fields vs. every `ScanDao` query's referenced columns vs. `@Database`'s
   registered entity list — consistent.
6. Retrofit DTOs vs. `server/schemas.py`'s Pydantic models, field by field. **One real gap found
   and fixed here**: `RegionDto` was missing `cnn_probabilities` (present in the server's
   `RegionOut`) — Gson would have silently dropped that field rather than failing to compile, so
   this was only catchable by an explicit side-by-side field diff, which is what found it.
7. Every `Screen` route in `Screen.kt` has exactly one matching `composable(...)` registration in
   `RiceGuardNavGraph.kt`, and vice versa (13/13).
8. Every clickable UI element wired to a real callback. **Two real bugs found and fixed here**:
   `HomeScreen`'s recent-scan rows and `HistoryScreen`'s history rows each had an `onClick`
   parameter that was declared but never attached to a `.clickable(...)` modifier — tapping them
   would have silently done nothing. Verified fixed by re-checking every `.clickable(` call site
   in the codebase (3/3 correctly pass `onClick = onClick`) plus a pattern search for the same
   "callback referenced but never invoked" shape elsewhere (none found).
9. CameraX capture → preview → Retake/Use Photo traced end to end: `ImageCapture.takePicture`
   only fires from the shutter button; `onPhotoCaptured` only ever sets local state (no network
   call); Retake deletes the local file and clears state before returning to a fresh camera
   binding.
10. Confirmed **exactly one call site** of the actual upload (`ScanRepository.submitScan`, via
    `ScanSessionViewModel.submitPhoto`), and it is inside `ProcessingScreen`'s `LaunchedEffect`,
    reachable only via the Photo Confirmation screen's "Use Photo" button — no path uploads a
    photo earlier than that.
11. Delete-one and delete-all traced end to end from their confirmation dialogs through to both
    the Room row and the on-disk image folder being removed.
12. Confirmed no hardcoded IP anywhere in Kotlin source (the one IP-shaped string in the codebase
    is a text-field *placeholder hint*, `"192.168.1.10"`, never used as an actual value); the
    default port (`8000`) is a user-overridable default, not a restriction.
13. Server route decorators (`server/main.py`) cross-checked against `ApiService.kt`'s Retrofit
    annotations — `/health`, `/predict` match; `/results/{request_id}/{filename}` is deliberately
    not a fixed Retrofit endpoint (URLs are server-generated and dynamic) but its URL construction
    (`ScanRepository.resolveUrl`) was traced to confirm it produces exactly that path shape.
14. Confirmed the resolved result-image URLs are the same absolute URLs used both for immediate
    display (Coil) and for the background history-save download — one code path, not two that
    could drift apart.
15. `ScanStatus.fromWire` mapping and `ProcessingScreen`'s branch to `LowConfidenceScreen` /
    `ModelDisagreementScreen` / `ResultScreen` traced against what the server actually populates
    for each status (`final_disease`/`final_confidence` vs. `yolo_prediction`+`cnn_prediction`).
16. Confirmed `/health` and `/predict` responses never include a raw server filesystem path —
    `PredictResponse` is built as an explicit Pydantic object (no field named `image` or any raw
    path exists on that model at all), with FastAPI's `response_model` as a second layer of
    filtering even if something extra were ever added.
17. Confirmed `server/main.py`'s upload cleanup runs inside a `finally` block wrapping the
    pipeline call, so it executes on every outcome (success, `ModelsNotReadyError`, or any other
    exception) — not just the success path.
18. Searched Gradle files, all Kotlin source, and the resource tree for any on-device ML
    dependency or bundled model file (TFLite/ONNX/PyTorch Mobile/ML Kit/`.pt`/`.tflite`/`.onnx`)
    — none exist. The Android app has no local inference capability of any kind.

No ML model was retrained, modified, or re-evaluated to perform this pass. `models/yolo_yolov8n/best.pt`
and `models/classifier_resnet50/best.pt` are unchanged (confirmed by file timestamp, matching every
prior check in this project's history).

## How an image actually flows, end to end

1. **Camera screen** (`ui/screens/camera/CameraScreen.kt`) — CameraX captures a JPEG to a
   private cache file (`ImageStorage.newCaptureStagingFile`). Nothing is sent yet.
2. **Photo Confirmation** — the file is shown full-screen; `ImageQualityChecker` (resolution/
   brightness/blur heuristics — *not* disease classification, see its own doc comment) runs in
   the background and shows non-blocking warnings. **Retake** deletes the local file and
   returns to Camera; only **Use Photo** proceeds.
3. **Processing** — `ScanSessionViewModel.submitPhoto()` calls `ScanRepository.submitScan()`,
   which POSTs the file to `/predict` as multipart form data.
4. **Server** (`server/main.py`) validates the upload, saves it to
   `server/tmp/<request_id>/upload.jpg`, and calls the real `RiceDiseasePipeline.run()` —
   serialized through a single-slot queue (`server/pipeline_service.py`'s `asyncio.Lock`, since
   there is one GPU) — then deletes the raw upload immediately and renders a detection-box
   visualization (`server/visualize.py`, since the pipeline itself only returns box coordinates,
   not an image) before returning structured JSON.
5. **Result / Explanation / Recommendation screens** read the JSON response (mapped to
   `domain/model/ScanResult`) and fetch the Grad-CAM/detection images directly from the
   server's `/results/<request_id>/<file>` URLs via Coil.
6. If **Save Scan History** is on (Settings, default ON — project brief section 26),
   `ScanRepository` downloads those same result images into this app's private storage
   (`context.filesDir/scans/<request_id>/`) and writes one `ScanRecord` row to Room, all in the
   background, *after* the result is already shown — the user never waits on this. If history
   is off, nothing is written to disk at all; the result is visible for the current session only
   (images load from the server's temporary URLs, which the server itself expires after ~30
   minutes — see `RESULT_TTL_SECONDS` in `server/pipeline_service.py`).

## History / data deletion

- **Delete one scan** (History Detail → trash icon → confirm): removes the Room row *and* its
  entire private-storage folder (`ImageStorage.deleteScan`) in one call — original, detection,
  and Grad-CAM images all go together, never partially.
- **Delete All Scan Data** (Settings → Privacy & Data): same, for every scan — `scanDao.deleteAll()`
  + `imageStorage.deleteAll()` (which removes `context.filesDir/scans/` entirely).
- Both require an explicit confirmation dialog naming exactly what will be removed; neither
  deletes on a single tap.

## Privacy

- The **original photo is never sent anywhere until the user taps "Use Photo."**
- The server **never permanently stores an uploaded image** — it's deleted from
  `server/tmp/` immediately after the pipeline finishes with it (`PipelineService.cleanup_upload`).
  Only the *derived* result images (Grad-CAM, detection box) persist briefly (≤30 min) so the
  phone can fetch them, then a background sweep removes them too.
- The server only accepts requests from private/local-network IPs
  (`server/security.py:LocalNetworkOnlyMiddleware`) — no cloud, no public internet destination,
  anywhere in this codebase.
- On the phone, history is opt-out-able (Save Scan History toggle) and fully, verifiably
  deletable (see above). Nothing is written to shared/public storage — only this app's private
  directory.

## Known limitations (disclosed deliberately, not hidden)

- **The underlying detector's recall is weak** (`PROJECT_README.md` §7.1) — most scans will
  legitimately take the "no detection → full-image CNN" path, which is expected, not a bug.
  The UI never implies a missed detection means "healthy."
- **Grad-CAM is an explanation aid, not proof of correctness** — the Explanation screen's
  copy says this explicitly (`PROJECT_README.md` §7.3 documents real cases where a correct,
  confident prediction's attention was not on the actual lesion).
- **Agreement between YOLO and the CNN does not guarantee correctness either** — both models
  can be confidently wrong together (`PROJECT_README.md` §7.4); this app surfaces
  `model_disagreement` only when they actually disagree, and does not claim more than that.
- **Treatment guidance is general agronomic information, not validated advice** —
  every disease's `treatment` field is marked `NEEDS_AUTHORITATIVE_VALIDATION` in the source
  knowledge base and is shown, unedited, in the Recommendation screen.
- **This is a local-network prototype**, not a production deployment — no authentication, no
  HTTPS (see the network security config's own comment for why cleartext is scoped and
  intentional here), no multi-user support. Explicitly out of scope per the brief.
