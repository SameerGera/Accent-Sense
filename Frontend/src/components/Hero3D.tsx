import { useRef, useEffect, useState } from 'react';

const MARKER_COLORS = ['#E879F9', '#8B5CF6', '#22D3EE'];

function makeWave(seed: number, n = 90): number[] {
  let s = seed;
  const rand = () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
  return Array.from({ length: n }, (_, i) => {
    const base = 0.25 + 0.6 * Math.abs(Math.sin(i * 0.18 + seed));
    return Math.max(0.08, Math.min(1, base * (0.6 + rand() * 0.8)));
  });
}

export default function Hero3D() {
  const ref = useRef<HTMLDivElement>(null);
  const [tilt, setTilt] = useState({ rx: -18, ry: 28 });
  const ringRefs = useRef<(HTMLDivElement | null)[]>([]);
  const wave = makeWave(7);

  useEffect(() => {
    let raf = 0;
    let t = 0;
    const loop = () => {
      t += 0.012;
      for (let i = 0; i < 3; i++) {
        const el = ringRefs.current[i];
        if (el) {
          el.style.transform = `rotateZ(${t * (20 + i * 10)}deg)`;
        }
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  const onMove = (e: React.MouseEvent) => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width - 0.5;
    const py = (e.clientY - r.top) / r.height - 0.5;
    setTilt({ rx: -18 + py * 14, ry: 28 + px * 22 });
  };

  const onLeave = () => setTilt({ rx: -18, ry: 28 });

  return (
    <div
      ref={ref}
      onMouseMove={onMove}
      onMouseLeave={onLeave}
      className="relative w-full h-[420px] sm:h-[460px] perspective-2000 select-none"
    >
      {/* ambient glow */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="w-72 h-72 rounded-full bg-fuchsia-500/20 blur-3xl animate-pulse-glow" />
      </div>

      {/* orbiting rings */}
      <div
        className="absolute left-1/2 top-1/2 preserve-3d"
        style={{
          transform: `translate(-50%,-50%) rotateX(${tilt.rx}deg) rotateY(${tilt.ry}deg)`,
          transformStyle: 'preserve-3d',
          transition: 'transform 0.2s ease-out',
        }}
      >
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            ref={(el) => {
              ringRefs.current[i] = el;
            }}
            className="absolute rounded-full border preserve-3d"
            style={{
              width: 220 + i * 90,
              height: 220 + i * 90,
              marginLeft: -(110 + i * 45),
              marginTop: -(110 + i * 45),
              borderColor: `${MARKER_COLORS[i]}55`,
              boxShadow: `0 0 40px ${MARKER_COLORS[i]}22`,
            }}
          >
            {/* node on ring */}
            <div
              className="absolute w-3 h-3 rounded-full"
              style={{
                left: '50%',
                top: -6,
                background: MARKER_COLORS[i],
                boxShadow: `0 0 16px ${MARKER_COLORS[i]}`,
                transform: 'translateX(-50%)',
              }}
            />
          </div>
        ))}

        {/* central floating card — the "prediction" */}
        <div
          className="absolute preserve-3d backface-hidden"
          style={{
            transform: `translate(-50%,-50%) translateZ(60px) rotateY(${-tilt.ry * 0.5}deg) rotateX(${-tilt.rx * 0.5}deg)`,
            transformStyle: 'preserve-3d',
          }}
        >
          <div className="w-56 sm:w-64 card-glass px-5 py-4 shadow-2xl">
            <div className="font-mono text-[10px] text-muted uppercase tracking-widest">Prediction</div>
            <div className="font-display text-xl font-bold text-fuchsia-500 mt-1">Telugu Influence</div>
            <div className="flex items-center gap-3 mt-3">
              <div className="flex-1 h-1.5 rounded-full bg-white/10 overflow-hidden">
                <div className="h-full rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500" style={{ width: '87%' }} />
              </div>
              <span className="font-mono text-xs text-muted">87%</span>
            </div>
            <div className="mt-3 flex gap-2">
              {MARKER_COLORS.map((c, i) => (
                <span key={i} className="w-2 h-2 rounded-full" style={{ background: c }} />
              ))}
            </div>
          </div>
        </div>

        {/* floating chips around the scene */}
        {[
          { label: '/ʈ/ retroflex', color: MARKER_COLORS[0], x: -150, y: -90, z: 80 },
          { label: 'vowel epenthesis', color: MARKER_COLORS[1], x: 160, y: -60, z: 40 },
          { label: 'syllable-timed', color: MARKER_COLORS[2], x: 130, y: 110, z: -30 },
          { label: 'schwa deletion', color: MARKER_COLORS[0], x: -140, y: 100, z: 20 },
        ].map((chip, i) => (
          <div
            key={i}
            className="absolute preserve-3d backface-hidden"
            style={{
              transform: `translate(${chip.x}px, ${chip.y}px) translateZ(${chip.z}px) rotateY(${-tilt.ry * 0.4}deg) rotateX(${-tilt.rx * 0.4}deg)`,
              transformStyle: 'preserve-3d',
            }}
          >
            <div
              className="px-3 py-1.5 rounded-lg text-[11px] font-mono whitespace-nowrap"
              style={{
                background: 'rgba(20,26,68,0.85)',
                border: `1px solid ${chip.color}66`,
                color: '#F1F0FB',
                boxShadow: `0 0 20px ${chip.color}33`,
              }}
            >
              {chip.label}
            </div>
          </div>
        ))}
      </div>

      {/* waveform strip at bottom */}
      <div className="absolute bottom-0 left-0 right-0 px-4">
        <div className="card-glass px-4 py-3 flex items-end gap-[2px] h-16 overflow-hidden">
          {wave.map((v, i) => (
            <div
              key={i}
              className="flex-1 rounded-sm transition-opacity"
              style={{
                height: `${v * 100}%`,
                background: `linear-gradient(to top, ${MARKER_COLORS[i % 3]}88, ${MARKER_COLORS[i % 3]})`,
                opacity: 0.6 + (i % 3) * 0.15,
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
