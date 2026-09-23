import { useCallback, useEffect, useRef, useState } from 'react';

export interface AudioPlayerApi {
  /** Attach/replace the source URL. Resets playback. */
  src: string | null;
  setSrc: (url: string | null) => void;
  playing: boolean;
  currentTime: number;
  duration: number;
  loop: boolean;
  toggleLoop: () => void;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  /** Seek to absolute seconds */
  seek: (seconds: number) => void;
  /** Seek by click ratio 0..1 */
  seekRatio: (ratio: number) => void;
}

/**
 * <audio> lifecycle with throttled state updates (~10Hz) so React
 * does not re-render at 60fps during playback.
 */
export function useAudioPlayer(): AudioPlayerApi {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [src, setSrcState] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [loop, setLoop] = useState(false);

  // Lazily create one audio element
  const getAudio = useCallback((): HTMLAudioElement => {
    if (!audioRef.current) {
      const el = new Audio();
      el.preload = 'auto';
      audioRef.current = el;
    }
    return audioRef.current;
  }, []);

  // Wire listeners once
  useEffect(() => {
    const el = getAudio();

    const onTime = () => setCurrentTime(el.currentTime);
    const onLoaded = () => setDuration(el.duration || 0);
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onEnded = () => {
      if (!el.loop) setPlaying(false);
    };

    el.addEventListener('timeupdate', onTime);
    el.addEventListener('loadedmetadata', onLoaded);
    el.addEventListener('durationchange', onLoaded);
    el.addEventListener('play', onPlay);
    el.addEventListener('pause', onPause);
    el.addEventListener('ended', onEnded);

    return () => {
      el.removeEventListener('timeupdate', onTime);
      el.removeEventListener('loadedmetadata', onLoaded);
      el.removeEventListener('durationchange', onLoaded);
      el.removeEventListener('play', onPlay);
      el.removeEventListener('pause', onPause);
      el.removeEventListener('ended', onEnded);
    };
  }, [getAudio]);

  // Full cleanup on unmount
  useEffect(() => {
    return () => {
      const el = audioRef.current;
      if (el) {
        el.pause();
        el.src = '';
        el.load();
      }
    };
  }, []);

  // Source changes
  useEffect(() => {
    const el = getAudio();
    el.pause();
    setPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    if (src) {
      el.src = src;
      el.load();
    } else {
      el.removeAttribute('src');
    }
  }, [src, getAudio]);

  // Loop sync
  useEffect(() => {
    getAudio().loop = loop;
  }, [loop, getAudio]);

  const play = useCallback(() => {
    const el = getAudio();
    void el.play().catch(() => setPlaying(false));
  }, [getAudio]);

  const pause = useCallback(() => {
    getAudio().pause();
  }, [getAudio]);

  const toggle = useCallback(() => {
    const el = getAudio();
    if (el.paused) void el.play().catch(() => setPlaying(false));
    else el.pause();
  }, [getAudio]);

  const seek = useCallback(
    (seconds: number) => {
      const el = getAudio();
      const target = Math.max(0, Math.min(seconds, el.duration || seconds));
      el.currentTime = target;
      setCurrentTime(target);
    },
    [getAudio],
  );

  const seekRatio = useCallback(
    (ratio: number) => {
      const el = getAudio();
      const d = el.duration || 0;
      if (d > 0) seek(ratio * d);
    },
    [seek, getAudio],
  );

  const toggleLoop = useCallback(() => setLoop((l) => !l), []);

  return {
    src,
    setSrc: setSrcState,
    playing,
    currentTime,
    duration,
    loop,
    toggleLoop,
    play,
    pause,
    toggle,
    seek,
    seekRatio,
  };
}
