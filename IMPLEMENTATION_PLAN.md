# Implementation Plan: Stitch Design Integration + Polish & Optimization

**Status:** In progress
**Date:** 2026-09-23
**Source design:** Stitch project `AccentSense Speech Analysis Interface` (ID `13670540037920493609`), screen `AccentSense — Acoustic Linguistics & Speech Analysis` (`163a970d19cb4d1fa5599472354cf13a`)
**Exported assets:** `stitch-export/accentsense-light.html`, `stitch-export/accentsense-light-screenshot.png`

**Goal:** Replace the current warm-Inter UI with Stitch's "Warm Editorial Minimalism" design, wire it to the real backend contract, and ship a polished, optimized, accessible result.

**Decisions (confirmed):**
- Explicit "Analyze speech" button (not auto-analyze on upload)
- Include ASR comparison section (`/api/downstream-asr`)
- Remove broken demo samples
- Material Symbols Outlined icons (drop `lucide-react`)

---

## Findings / bugs to fix during integration

1. **API contract mismatch** — `LandingPage.tsx` expects `predicted_accent`/`probabilities`/`audio_duration_sec`/`model_used`/`model_type`; backend returns `predicted_influence`/`language_family`/`all_scores`/`salient_regions`/`timestamps`/`saliency_curve`/`is_mock_prototype`. Results render broken data today.
2. **Demo samples broken** — frontend calls `GET /api/demo/*`; no such endpoint and no `.wav` files exist in the repo.
3. **`lucide-react` installed but never imported** — remove dependency.
4. **Rich backend data unused** — `salient_regions` maps onto Stitch "Why this result?" + "Audio breakdown"; `/api/downstream-asr` has no UI.

---

## Phase 1 — Design tokens (foundation)

| File | Change |
|------|--------|
| `Frontend/index.html` | Google Fonts: preconnect + **Newsreader** (400/500) + **Plus Jakarta Sans** (400/500/600) + **Material Symbols Outlined** (`display=swap`); `theme-color` `#fff8f4` / dark `#32302e`; trimmed OG meta |
| `Frontend/tailwind.config.js` | Stitch semantic palette (`surface*`, `on-surface`, `primary`, `outline*`, `inverse-*`, `error*`), typography scale (`display-lg` → `label-sm`), spacing tokens (`space-*`, `gutter*`, `margin*`), restrained radii (4px default / 8px max), `fontFamily`: headline → Newsreader, body/label → Plus Jakarta Sans |
| `Frontend/src/index.css` | CSS variables `:root` (light) + `.dark` (exact Stitch hexes); restyle `btn-primary`/`btn-secondary`/`input-field`/tags (34px height, 4px radius, hairline borders); `focus-visible`: 2px offset primary outline; `@media print` for PDF export; `@media (prefers-reduced-motion)`; `.material-symbols-outlined` rules; remove dead classes (`glow-line`, superseded waveform idle) |

### Stitch tokens (reference)

**Light:** canvas `#FBF9F5`/`#fff8f4`, surface-1 `#F5F2EB`, surface-2 `#ECE7DD`, text `#1C1A18`, text-2 `#68625B`, subtle `#8F8880`, hairline `#E5DFD5`, faint `#EDE8E0`, accent `#B85028`, accent-tint `#F4EBE6`.
M3-named (used by exported HTML): `primary #983912`, `primary-container #b85028`, `on-primary #ffffff`, `on-surface #1d1b19`, `on-surface-variant #57423b`, `outline #8a726a`, `outline-variant #ddc0b7`, `surface #fff8f4`, `surface-container #f3ede9`, `surface-container-low #f9f2ef`, `surface-container-high #ede7e3`, `surface-container-highest #e7e1de`, `secondary #635d57`, `error #ba1a1a`, `error-container #ffdad6`.

**Dark:** canvas `#141312`, surface-1 `#1D1B1A`, surface-2 `#272422`, text `#EDEAE4`, text-2 `#A8A29A`, subtle `#85807A`, hairline `#2E2B27`, accent `#D96B43`, tint `#291D17`.
M3: `inverse-surface #32302e`, `inverse-on-surface #f6efec`, `inverse-primary #ffb59c`.

**Typography:** Newsreader (display/headlines, weight 400 mostly), Plus Jakarta Sans (body 400, labels 500/600).
Scale: display-lg 48/56 -0.02em · display-lg-mobile 34/42 · headline-lg 32/40 · headline-md 24/32 500 · headline-sm 18/26 600 (Jakarta) · body-lg 17/28 · body-md 15/24 · body-sm 13/20 · label-md 13/18 500 · label-sm 11/16 600 0.06em uppercase · label-code 12/16 500.

**Spacing:** space-xs .25rem · space-sm .5rem · space-md 1rem · space-lg 1.5rem · space-xl 2.5rem · gutter-sm 1rem · gutter 1.5rem · gutter-lg 2rem · margin-sm 1rem · margin 2rem · margin-lg 3.5rem.

**Radius:** DEFAULT 0.125rem (2px) · lg 0.25rem (4px) · xl 0.5rem (8px) · full 0.75rem. Buttons/inputs 4px; badges 2px; no pills; no shadows (tonal tiers + 1px hairlines only).

---

## Phase 2 — Fix the API contract

| File | Change |
|------|--------|
| `src/lib/types.ts` **(new)** | `PredictionResponse`, `SalientRegion`, `DownstreamASRResponse` mirroring Pydantic models |
| `src/lib/api.ts` **(new)** | `fetchPredict(file)` → `POST /api/predict`; `fetchASR(accent, text?)` → `POST /api/downstream-asr`; typed errors |
| `src/lib/format.ts` **(new)** | `formatTime`, `formatPercent`, `accentDisplayName` (`West_Midlands` → `West Midlands`) |

---

## Phase 3 — Hooks

| File | Change |
|------|--------|
| `hooks/useAudioPlayer.ts` **(new)** | `<audio>` lifecycle: play/pause, throttled current-time (~10Hz), seek, loop toggle, duration, cleanup |
| `hooks/useWaveform.ts` **(new)** | `decodeAudioData` → ~120 peak bars, memoized; debounced `ResizeObserver`; DPR-aware; exposes redraw for theme changes |
| `hooks/usePrediction.ts` **(new)** | `idle → loading → success | error`; `analyze()`, `reset()` wrapping `fetchPredict` |
| `ThemeProvider.tsx` | Keep (localStorage + system preference + `.dark` class) |
| `ThemeToggle.tsx` | Restyle: bordered Stitch button, Material icon + "Dark"/"Light" label |

---

## Phase 4 — Components (bottom-up)

| Component | Content |
|-----------|---------|
| `WaveformCanvas.tsx` | Canvas peaks (unplayed `#8a726a` / dark `#57423b`, played `#b85028`), click-to-seek, playhead (1.5px line + 6px cap), optional saliency overlay post-prediction, redraw on theme toggle |
| `AudioPlayer.tsx` | Play/pause, title+duration, timecode, `WaveformCanvas`, loop toggle, **"Analyze speech →"** CTA (disabled until file) |
| `UploadStage.tsx` | Heading + format hint; dashed drop zone (`graphic_eq`, browse label, status dot + `Ready for input` / `Loaded: name (XX KB)`); **mic recording** (MediaRecorder, timer, stop → auto-load); accept wav/mp3/ogg/webm/m4a/flac |
| `AnalysisProgress.tsx` | Pulsing `model_training` icon, "Analysing acoustic features", 2px hairline progress bar (0→100, completes on response) |
| `RegionInspector.tsx` | "Audio breakdown" clickable cards from `salient_regions` (part #, timestamps, phenomenon, explanation); click seeks to `start_time_sec` |
| `ResultsDossier.tsx` | "LIKELY ACCENT" label → `predicted_influence` + `language_family` + big confidence %; "Other possibilities" from `all_scores`; 7-class probability rows; **"Why this result?"** numbered rows from `salient_regions`; `<RegionInspector/>`; Background / How-it-works notes; honest `is_mock_prototype` badge; Try another recording / Download JSON / Save PDF (`window.print()`) |
| `ASRComparison.tsx` **(new)** | "Speech recognition": baseline vs accent-adapted transcripts, strategy, phonetic corrections; own loading/error states |
| `Header.tsx` | Sticky 60px: Newsreader wordmark, How it works / About anchors, `ThemeToggle`, "Start over" |
| `EditorialIntro.tsx` | Kicker `SPEECH ANALYSIS`; headline adapted to accent domain ("Find out what your English accent reveals about where you're from."); plain-English subhead → 7 UK regional accents |
| `Footer.tsx` | Wordmark, ©, How it works / Privacy / Contact, hairline top |
| `LandingPage.tsx` | Thin orchestrator: composes sections; owns file/prediction/loading/error/recording state |

**Delete:** `Hero3D.tsx`. **Remove:** `lucide-react` from `package.json`.

---

## Phase 5 — Polish & optimize

**Performance**
- Waveform peaks once per file, memoized; fallback if `decodeAudioData` fails: `<audio>` duration + procedural envelope
- Playhead rAF/CSS-driven, ~4–10Hz React state; debounced canvas redraw (resize + theme)
- `useMemo` sorted scores; `useCallback` handlers; full cleanup (object URLs, audio, rAF, intervals)
- Remove dead code/deps; limit font weights; report bundle size before/after

**Accessibility**
- `aria-label`s; `role="progressbar"` + `aria-valuenow`; results `aria-live="polite"`
- Drop zone keyboard-operable; waveform arrow-key seek when focused
- `focus-visible`: 2px offset primary outline
- Contrast AA verified; `prefers-reduced-motion` disables animations

**Responsive** (Stitch designMd): Desktop ≥1024 12-col/2rem gutters/3.5rem margins/max-1280 · Tablet 768–1023 8-col · Mobile <768 single-col/1rem/shorter waveform/stacked bars.

**Dark mode:** inverse-* tokens; waveform redraw on toggle; persistence + OS preference verified.

**Print/PDF:** hide nav/footer/buttons; expand results; page-break avoid; black-on-white.

**Error states:** `error-container` palette, dismissible, backend-unreachable + retry.

**Copy:** plain English only; no "AI-powered/unlock/insights"; accent framing (not native-language); mock mode honestly labeled.

---

## Phase 6 — Verification

1. `npm run typecheck` — clean
2. `npm run lint` — clean
3. `npm run build` — success; record bundle sizes
4. Manual QA (dev server + backend :8000): upload → waveform → Analyze → progress → correct backend fields in results · mic record → auto-load → analyze · ASR populates · theme toggle full-page + canvas + persistence + OS pref · JSON download · print preview · breakpoints 375/768/1024/1440 · keyboard-only + reduced-motion
5. Grep stale tokens: `fuchsia|violet|cyan|glow|lucide|predicted_accent|probabilities|/api/demo`

---

## Execution order

1. Design tokens (`index.html`, `tailwind.config.js`, `index.css`)
2. `lib/` types + api + format (contract fix)
3. Hooks (`useAudioPlayer`, `useWaveform`, `usePrediction`)
4. Components bottom-up (WaveformCanvas → AudioPlayer → UploadStage → AnalysisProgress → RegionInspector → ResultsDossier → ASRComparison → Header → EditorialIntro → Footer → LandingPage)
5. Delete Hero3D, remove lucide-react
6. Polish pass (a11y, print, reduced-motion, perf)
7. Full verification suite

## Risks

- `decodeAudioData` may fail on some webm/mp3 → fallback procedural envelope (degraded, labeled in code)
- Material Symbols render-blocking → `display=swap` + preconnect
- Whisper ASR may be slow → lazy ASR section with own loading state; results render first
- No trained checkpoint yet → `is_mock_prototype: true` path must look honest, not fake-precise
