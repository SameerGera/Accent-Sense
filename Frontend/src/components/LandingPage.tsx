import { AnimatePresence, motion } from 'framer-motion';
import { scaleIn, stagger } from '../lib/motion';
import { Header } from './Header';
import { EditorialIntro } from './EditorialIntro';
import { UploadStage } from './UploadStage';
import { AudioPlayer } from './AudioPlayer';
import { AnalysisProgress } from './AnalysisProgress';
import { ResultsDossier } from './ResultsDossier';
import { ASRComparison } from './ASRComparison';
import { HowItWorks } from './HowItWorks';
import { Footer } from './Footer';
import { useAudioPlayer } from '../hooks/useAudioPlayer';
import { useWaveform } from '../hooks/useWaveform';
import { useASR, usePrediction } from '../hooks/usePrediction';
import { useCallback, useEffect, useRef, useState } from 'react';

export default function LandingPage() {
  // File / recording state
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [recTime, setRecTime] = useState(0);
  const [micError, setMicError] = useState<string | null>(null);
  const [errorHidden, setErrorHidden] = useState(false);

  // Hooks — destructure stable callbacks so identities don't churn per render
  const player = useAudioPlayer();
  const waveform = useWaveform();
  const pred = usePrediction();
  const asr = useASR();
  const {
    reset: resetPred,
    analyze,
    status: predStatus,
    prediction,
    error: predError,
    progress,
  } = pred;
  const { load: loadAsr, clear: clearAsr } = asr;
  const { clear: clearWaveform, load: loadWaveform, peaks } = waveform;
  const { setSrc, pause, seek, duration } = player;

  // Recording refs
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const asrLoadedKey = useRef<string | null>(null);
  const previewUrlRef = useRef<string | null>(null);

  // Keep preview URL in ref for safe cleanup
  useEffect(() => {
    previewUrlRef.current = previewUrl;
  }, [previewUrl]);

  // Unhide error banner whenever prediction status changes
  useEffect(() => {
    setErrorHidden(false);
  }, [predStatus]);

  // Trigger ASR comparison once per successful prediction
  useEffect(() => {
    if (predStatus === 'success' && prediction) {
      const key = `${prediction.predicted_influence}:${prediction.confidence}`;
      if (asrLoadedKey.current !== key) {
        asrLoadedKey.current = key;
        void loadAsr(prediction.predicted_influence);
      }
    }
  }, [predStatus, prediction, loadAsr]);

  // Cleanup on unmount: media stream, timers, object URLs
  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
      mediaRef.current?.stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  /** Load a new file into player + waveform, clear prior results. */
  const onFile = useCallback(
    (f: File) => {
      setMicError(null);
      setErrorHidden(false);
      resetPred();
      clearAsr();
      asrLoadedKey.current = null;
      clearWaveform();

      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
      const url = URL.createObjectURL(f);
      setFile(f);
      setPreviewUrl(url);
      setSrc(url);
      loadWaveform(url);
    },
    [resetPred, clearAsr, clearWaveform, setSrc, loadWaveform],
  );

  /** Explicit "Analyze speech" — pauses playback, runs prediction. */
  const handleAnalyze = useCallback(() => {
    if (!file) return;
    pause();
    setErrorHidden(false);
    void analyze(file);
  }, [file, pause, analyze]);

  /** Full reset — new session. */
  const startOver = useCallback(() => {
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (mediaRef.current && mediaRef.current.state !== 'inactive') {
      mediaRef.current.stop();
    }
    setRecording(false);
    setRecTime(0);
    setMicError(null);
    setErrorHidden(false);

    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    setFile(null);
    setPreviewUrl(null);

    setSrc(null);
    clearWaveform();
    resetPred();
    clearAsr();
    asrLoadedKey.current = null;

    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [setSrc, clearWaveform, resetPred, clearAsr]);

  /** Mic recording */
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mr.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        const f = new File([blob], `recording-${Date.now()}.webm`, {
          type: 'audio/webm',
        });
        onFile(f);
      };
      mediaRef.current = mr;
      mr.start();
      setRecording(true);
      setRecTime(0);
      timerRef.current = window.setInterval(() => setRecTime((t) => t + 1), 1000);
    } catch {
      setMicError('Microphone access denied. Allow the microphone and try again.');
    }
  }, [onFile]);

  const stopRecording = useCallback(() => {
    if (mediaRef.current && mediaRef.current.state !== 'inactive') {
      mediaRef.current.stop();
    }
    setRecording(false);
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  /** Seek helper for region cards */
  const seekTo = useCallback((seconds: number) => seek(seconds), [seek]);

  const showError = predStatus === 'error' && predError && !errorHidden;
  const saliency =
    prediction && duration > 0
      ? {
          timestamps: prediction.timestamps,
          curve: prediction.saliency_curve,
          duration,
        }
      : null;

  const stageKey = file && !recording ? 'player' : recording ? 'recording' : 'upload';

  return (
    <div
      id="top"
      className="min-h-screen flex flex-col print-full bg-surface text-on-surface"
    >
      <Header
        onStartOver={startOver}
        canStartOver={!!file || recording || predStatus !== 'idle'}
      />

      <motion.main
        className="flex-grow max-w-3xl mx-auto w-full px-space-lg md:px-margin-lg py-space-xl space-y-space-xl print-full"
        variants={stagger(0, 0.12)}
        initial="hidden"
        animate="show"
      >
        <EditorialIntro />

        {/* Input stage: upload / recording OR player — animated swap */}
        <section className="space-y-space-lg border-t border-outline-variant pt-space-xl" id="inputStage" aria-label="Audio input">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={stageKey}
              variants={scaleIn}
              initial="hidden"
              animate="show"
              exit="exit"
              className="space-y-space-lg"
            >
              {!file && (
                <UploadStage
                  onFile={onFile}
                  recording={recording}
                  recTime={recTime}
                  onStartRecording={startRecording}
                  onStopRecording={stopRecording}
                  error={micError}
                />
              )}
              {file && !recording && (
                <AudioPlayer
                  player={player}
                  peaks={peaks}
                  fileName={file.name}
                  saliency={saliency}
                  onAnalyze={handleAnalyze}
                  analyzing={predStatus === 'loading'}
                  canAnalyze={predStatus !== 'loading'}
                />
              )}
            </motion.div>
          </AnimatePresence>
        </section>

        {/* Analysis progress */}
        <AnimatePresence mode="wait">
          {predStatus === 'loading' && (
            <motion.div
              key="progress"
              variants={scaleIn}
              initial="hidden"
              animate="show"
              exit="exit"
            >
              <AnalysisProgress progress={progress} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error banner */}
        <AnimatePresence>
          {showError && (
            <motion.div
              className="p-space-md flex items-start justify-between gap-space-md no-print bg-error-container border border-error rounded text-on-error-container"
              role="alert"
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.3 }}
            >
              <div className="flex items-start gap-space-sm">
                <span className="material-symbols-outlined !text-[18px] mt-0.5" aria-hidden="true">
                  error
                </span>
                <span className="font-body-sm text-body-sm">{predError}</span>
              </div>
              <div className="flex items-center gap-space-sm shrink-0">
                <button
                  type="button"
                  onClick={handleAnalyze}
                  className="btn-secondary !h-[28px] !text-label-code"
                >
                  Retry
                </button>
                <button
                  type="button"
                  onClick={() => setErrorHidden(true)}
                  className="btn-tertiary !h-[28px] !text-label-code"
                  aria-label="Dismiss error"
                >
                  Dismiss
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results dossier + ASR */}
        <AnimatePresence mode="wait">
          {predStatus === 'success' && prediction && (
            <motion.div
              key="results"
              className="space-y-space-xl"
              variants={{
                hidden: { opacity: 0 },
                show: {
                  opacity: 1,
                  transition: { duration: 0.35, staggerChildren: 0.12 },
                },
                exit: { opacity: 0, transition: { duration: 0.2 } },
              }}
              initial="hidden"
              animate="show"
              exit="exit"
            >
              <motion.div
                variants={{
                  hidden: { opacity: 0, y: 12 },
                  show: {
                    opacity: 1,
                    y: 0,
                    transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1] },
                  },
                }}
              >
                <ResultsDossier
                  prediction={prediction}
                  duration={duration}
                  onSeek={seekTo}
                  onReset={startOver}
                />
              </motion.div>
              <motion.div
                variants={{
                  hidden: { opacity: 0, y: 10 },
                  show: {
                    opacity: 1,
                    y: 0,
                    transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1], delay: 0.15 },
                  },
                }}
              >
                <ASRComparison
                  asr={asr}
                  accent={prediction.predicted_influence}
                  onRetry={() => void loadAsr(prediction.predicted_influence)}
                />
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Always-visible explainer (nav anchor) */}
        <HowItWorks />
      </motion.main>

      <Footer />
    </div>
  );
}
