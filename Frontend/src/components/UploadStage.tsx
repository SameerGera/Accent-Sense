import { useCallback, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { fadeRise, scaleIn } from '../lib/motion';

export interface UploadStageProps {
  onFile: (file: File) => void;
  recording: boolean;
  recTime: number;
  onStartRecording: () => void;
  onStopRecording: () => void;
  error: string | null;
}

const ACCEPTED =
  'audio/wav,audio/mpeg,audio/mp3,audio/ogg,audio/webm,audio/mp4,audio/x-m4a,audio/flac,.wav,.mp3,.ogg,.webm,.m4a,.flac';

function formatTime(s: number): string {
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

export function UploadStage({
  onFile,
  recording,
  recTime,
  onStartRecording,
  onStopRecording,
  error,
}: UploadStageProps) {
  const [dragOver, setDragOver] = useState(false);
  const [status, setStatus] = useState('Ready for input');
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (f: File) => {
      setStatus(`Loaded: ${f.name.substring(0, 24)} (${(f.size / 1024).toFixed(0)} KB)`);
      onFile(f);
    },
    [onFile],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile],
  );

  if (recording) {
    return (
      <motion.div
        className="w-full p-space-lg text-center space-y-space-md bg-surface-container border border-outline-variant rounded"
        aria-live="polite"
        initial={{ opacity: 0, scale: 0.96 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.97 }}
        transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      >
        <div className="flex items-center justify-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full rec-pulse bg-error" />
          <span className="font-label-code text-label-code tabular">{formatTime(recTime)}</span>
          <span className="font-body-sm text-body-sm text-on-surface-variant">
            Recording
          </span>
        </div>
        {/* Live waveform hint */}
        <div className="flex items-end justify-center gap-1 h-8" aria-hidden="true">
          {Array.from({ length: 24 }).map((_, i) => (
            <div
              key={i}
              className="w-1 rounded-sm bg-primary-container"
              style={{
                height: '100%',
                opacity: 0.55,
                transformOrigin: 'bottom',
                animation: `rec-bar 1.4s ease-in-out ${i * 0.07}s infinite`,
              }}
            />
          ))}
        </div>
        <button type="button" onClick={onStopRecording} className="btn-primary mx-auto">
          <span className="material-symbols-outlined !text-[18px]" aria-hidden="true">
            stop
          </span>
          <span>Stop recording</span>
        </button>
        <style>{`
          @keyframes rec-bar {
            0%, 100% { transform: scaleY(0.25); }
            50% { transform: scaleY(0.85); }
          }
        `}</style>
      </motion.div>
    );
  }

  return (
    <motion.div
      className="w-full space-y-space-md"
      variants={fadeRise}
      initial="hidden"
      animate="show"
      exit="exit"
    >
      {/* Section heading */}
      <motion.div className="text-center space-y-space-xs" variants={fadeRise}>
        <h2 className="font-sans text-headline-sm text-on-surface">
          Upload audio
        </h2>
        <p className="font-body-sm text-body-sm text-on-surface-variant">
          Speak naturally in English for a few seconds · WAV, MP3, OGG, M4A, or WebM (up to 30s)
        </p>
      </motion.div>

      {/* Drop zone — height closer to player card for stage symmetry */}
      <motion.div
        className={`drop-zone py-space-lg px-space-xl space-y-space-sm ${dragOver ? 'drag-over' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => fileRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            fileRef.current?.click();
          }
        }}
        role="button"
        tabIndex={0}
        aria-label="Upload an audio file: drop a file here or press Enter to browse"
        variants={scaleIn}
        whileHover={{ y: -2 }}
        whileTap={{ scale: 0.99 }}
        transition={{ type: 'spring', stiffness: 400, damping: 28 }}
      >
        <input
          ref={fileRef}
          type="file"
          accept={ACCEPTED}
          className="sr-only"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
            e.target.value = '';
          }}
        />
        <motion.span
          className="material-symbols-outlined !text-[36px] text-primary-container"
          aria-hidden="true"
          animate={
            dragOver
              ? { scale: 1.12, rotate: [0, -6, 6, 0] }
              : { scale: 1, rotate: 0 }
          }
          transition={{ duration: 0.35 }}
        >
          graphic_eq
        </motion.span>
        <div className="space-y-space-xs">
          <p className="font-body-md text-body-md font-medium text-on-surface">
            Drop an audio file here, or{' '}
            <span
              className="cursor-pointer hover:underline text-primary-container"
            >
              browse files
            </span>
          </p>
          <p className="font-body-sm text-body-sm text-on-surface-variant">
            Sent securely to the analysis server. Recordings are never stored.
          </p>
        </div>
        <div className="flex items-center gap-space-sm pt-space-xs">
          <motion.span
            className="status-dot"
            animate={{ scale: [1, 1.3, 1] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          />
          <span className="font-label-code text-label-code text-on-surface-variant">
            {status}
          </span>
        </div>
      </motion.div>

      {/* Mic record */}
      <motion.button
        type="button"
        onClick={onStartRecording}
        className="btn-secondary w-full group"
        whileHover={{ y: -1 }}
        whileTap={{ scale: 0.98 }}
        transition={{ type: 'spring', stiffness: 500, damping: 30 }}
      >
        <span className="material-symbols-outlined !text-[18px] transition-transform duration-200 group-hover:scale-110" aria-hidden="true">
          mic
        </span>
        <span>Record with microphone</span>
      </motion.button>

      {error && (
        <motion.p
          className="font-body-sm text-body-sm text-center text-error"
          role="alert"
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
        >
          {error}
        </motion.p>
      )}
    </motion.div>
  );
}
