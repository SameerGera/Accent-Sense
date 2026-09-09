import { useState, useEffect, useRef } from 'react';
import {
  Mic, Upload, Play, Loader2, ChevronDown, Info, MapPin, Sparkles,
  Activity, Brain, Globe2, ShieldCheck, Zap, Ear, ArrowRight,
} from 'lucide-react';
import Hero3D from '@/components/Hero3D';

const MARKER_COLORS = ['#E879F9', '#8B5CF6', '#22D3EE'];

const SAMPLES = [
  {
    id: 'telugu',
    label: 'Sample A — 14s clip',
    speaker: 'Speaker 07, Hyderabad',
    prediction: 'Telugu Influence',
    confidence: 87,
    alternatives: [
      { lang: 'Kannada Influence', score: 8 },
      { lang: 'Tamil Influence', score: 5 },
    ],
    markers: [
      { start: 6, end: 20, label: 'retroflex /ʈ/ realization', note: '“t” sounds pulled back on tongue' },
      { start: 38, end: 52, label: 'syllable-timed rhythm', note: 'even stress across syllables' },
      { start: 68, end: 80, label: 'vowel epenthesis', note: 'short vowel inserted in consonant clusters' },
    ],
    region: 'Telangana / Andhra Pradesh',
    regionNote: 'Telugu-majority belt, southern India',
  },
  {
    id: 'punjabi',
    label: 'Sample B — 11s clip',
    speaker: 'Speaker 22, Ludhiana',
    prediction: 'Punjabi Influence',
    confidence: 79,
    alternatives: [
      { lang: 'Hindi Influence', score: 13 },
      { lang: 'Urdu Influence', score: 6 },
    ],
    markers: [
      { start: 4, end: 16, label: 'tonal pitch accent', note: 'residual tone on nasal-adjacent vowels' },
      { start: 30, end: 44, label: 'word-final schwa deletion', note: 'dropped ending vowel' },
      { start: 58, end: 70, label: 'geminate consonants', note: 'doubled consonant length' },
    ],
    region: 'Punjab',
    regionNote: 'Northwestern India',
  },
  {
    id: 'bengali',
    label: 'Sample C — 17s clip',
    speaker: 'Speaker 41, Kolkata',
    prediction: 'Bengali Influence',
    confidence: 91,
    alternatives: [
      { lang: 'Odia Influence', score: 4 },
      { lang: 'Assamese Influence', score: 3 },
    ],
    markers: [
      { start: 8, end: 22, label: '/v/–/b/ merger', note: '“v” realized closer to “b”' },
      { start: 34, end: 48, label: 'unaspirated stops', note: 'reduced aspiration on “p/t/k”' },
      { start: 60, end: 76, label: 'diphthong flattening', note: '“ow” sounds shortened' },
    ],
    region: 'West Bengal',
    regionNote: 'Eastern India',
  },
] as const;

type Sample = typeof SAMPLES[number];
type Marker = Sample['markers'][number];
type Alt = Sample['alternatives'][number];

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

const FEATURES = [
  {
    icon: Brain,
    title: 'Explainable AI',
    desc: 'Every prediction comes with the phonetic markers that drove it — no black boxes.',
    color: '#E879F9',
  },
  {
    icon: Activity,
    title: 'Acoustic feature analysis',
    desc: 'Retroflex consonants, vowel epenthesis, rhythm patterns, and more — mapped to timestamps.',
    color: '#8B5CF6',
  },
  {
    icon: Globe2,
    title: 'Regional mapping',
    desc: 'Pin a speaker’s likely native-language region on the map with confidence intervals.',
    color: '#22D3EE',
  },
  {
    icon: ShieldCheck,
    title: 'Private by design',
    desc: 'Audio is processed for inference and discarded. No biometric storage, ever.',
    color: '#34D399',
  },
  {
    icon: Zap,
    title: 'Real-time inference',
    desc: 'Sub-second classification on short clips, with streaming alternatives as audio plays.',
    color: '#FBBF24',
  },
  {
    icon: Ear,
    title: 'Built for linguists',
    desc: 'IPA-aligned markers and phonological notes, readable by speech researchers and learners alike.',
    color: '#F472B6',
  },
];

const STEPS = [
  { n: '01', title: 'Capture speech', desc: 'Record live or upload a short audio clip of the speaker.' },
  { n: '02', title: 'Extract features', desc: 'The model isolates phonetic, prosodic, and rhythmic features from the waveform.' },
  { n: '03', title: 'Classify & explain', desc: 'A native-language influence prediction is returned with the markers that justify it.' },
];

export default function LandingPage() {
  const [selectedId, setSelectedId] = useState<string>(SAMPLES[0].id);
  const [status, setStatus] = useState<'idle' | 'analyzing' | 'done'>('idle');
  const [result, setResult] = useState<Sample | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const selected = SAMPLES.find((s) => s.id === selectedId) as Sample;
  const wave = makeWave(selectedId.length + selectedId.charCodeAt(0));

  useEffect(() => () => { if (timeoutRef.current) clearTimeout(timeoutRef.current); }, []);

  const analyze = () => {
    setStatus('analyzing');
    setResult(null);
    timeoutRef.current = setTimeout(() => {
      setResult(selected);
      setStatus('done');
    }, 1400);
  };

  const reset = (id: string) => {
    setSelectedId(id);
    setStatus('idle');
    setResult(null);
    setPickerOpen(false);
  };

  return (
    <div className="min-h-screen bg-ink-900 text-[#F1F0FB] font-sans">
      {/* Nav */}
      <nav className="sticky top-0 z-50 backdrop-blur-xl bg-ink-900/70 border-b border-white/5">
        <div className="max-w-6xl mx-auto px-5 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center">
              <Ear size={18} className="text-ink-900" />
            </div>
            <span className="font-display text-lg font-extrabold tracking-tight">AccentSense</span>
          </div>
          <div className="hidden md:flex items-center gap-7 text-sm text-muted">
            <a href="#features" className="hover:text-white transition-colors">Features</a>
            <a href="#how" className="hover:text-white transition-colors">How it works</a>
            <a href="#demo" className="hover:text-white transition-colors">Live demo</a>
          </div>
          <a href="#demo" className="btn-primary px-4 py-2 text-sm flex items-center gap-1.5">
            Try demo <ArrowRight size={14} />
          </a>
        </div>
      </nav>

      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-10 -left-20 w-80 h-80 rounded-full bg-violet-600/20 blur-3xl" />
          <div className="absolute top-40 right-0 w-96 h-96 rounded-full bg-fuchsia-500/15 blur-3xl" />
          <div className="absolute bottom-0 left-1/3 w-72 h-72 rounded-full bg-cyan-500/10 blur-3xl" />
        </div>

        <div className="relative max-w-6xl mx-auto px-5 pt-16 pb-10 grid lg:grid-cols-2 gap-8 items-center">
          <div className="animate-rise">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-fuchsia-500/30 bg-fuchsia-500/10 text-xs font-mono text-fuchsia-400 mb-6">
              <Sparkles size={12} /> explainable native-language influence detection
            </div>
            <h1 className="font-display text-4xl sm:text-5xl lg:text-6xl font-extrabold leading-[1.05] tracking-tight">
              Hear where a voice<br />
              <span className="gradient-text">comes from.</span>
            </h1>
            <p className="mt-5 text-lg text-muted max-w-md leading-relaxed">
              AccentSense detects the native-language influence in spoken audio — and shows you the exact phonetic markers behind every prediction.
            </p>
            <div className="mt-7 flex flex-wrap gap-3">
              <a href="#demo" className="btn-primary px-6 py-3 text-sm flex items-center gap-2">
                <Play size={15} /> Try the live demo
              </a>
              <a href="#features" className="btn-ghost px-6 py-3 text-sm flex items-center gap-2">
                <Info size={15} /> See how it works
              </a>
            </div>
            <div className="mt-8 flex items-center gap-6 text-xs text-muted font-mono">
              <span>3 sample accents</span>
              <span className="w-1 h-1 rounded-full bg-muted/40" />
              <span>IPA-aligned markers</span>
              <span className="w-1 h-1 rounded-full bg-muted/40" />
              <span>No signup needed</span>
            </div>
          </div>

          <div className="animate-rise" style={{ animationDelay: '0.15s' }}>
            <Hero3D />
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="max-w-6xl mx-auto px-5 py-20">
        <div className="text-center mb-12">
          <div className="font-mono text-xs text-fuchsia-400 uppercase tracking-widest mb-3">Capabilities</div>
          <h2 className="font-display text-3xl sm:text-4xl font-bold">Not a black box. A reasoning partner.</h2>
          <p className="mt-3 text-muted max-w-xl mx-auto">AccentSense pairs every prediction with the phonetic evidence behind it, so you can trust — and audit — the result.</p>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {FEATURES.map((f) => {
            const Icon = f.icon;
            return (
              <div key={f.title} className="card p-6 transition-all duration-300 hover:-translate-y-1 hover:border-fuchsia-500/40 group">
                <div className="w-11 h-11 rounded-xl flex items-center justify-center mb-4 transition-transform group-hover:scale-110" style={{ background: `${f.color}1a`, border: `1px solid ${f.color}40` }}>
                  <Icon size={20} style={{ color: f.color }} />
                </div>
                <h3 className="font-display text-lg font-bold mb-1.5">{f.title}</h3>
                <p className="text-sm text-muted leading-relaxed">{f.desc}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* How it works */}
      <section id="how" className="relative py-20 border-y border-white/5 bg-ink-800/50">
        <div className="max-w-5xl mx-auto px-5">
          <div className="text-center mb-14">
            <div className="font-mono text-xs text-cyan-400 uppercase tracking-widest mb-3">Pipeline</div>
            <h2 className="font-display text-3xl sm:text-4xl font-bold">From waveform to explanation</h2>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            {STEPS.map((s, i) => (
              <div key={s.n} className="relative">
                <div className="font-display text-5xl font-extrabold text-white/10 mb-3">{s.n}</div>
                <h3 className="font-display text-xl font-bold mb-2">{s.title}</h3>
                <p className="text-sm text-muted leading-relaxed">{s.desc}</p>
                {i < STEPS.length - 1 && (
                  <div className="hidden md:block absolute top-8 -right-3 text-fuchsia-500/40">
                    <ArrowRight size={20} />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Demo */}
      <section id="demo" className="max-w-6xl mx-auto px-5 py-20">
        <div className="text-center mb-10">
          <div className="font-mono text-xs text-fuchsia-400 uppercase tracking-widest mb-3">Interactive</div>
          <h2 className="font-display text-3xl sm:text-4xl font-bold">Try it yourself</h2>
          <p className="mt-3 text-muted max-w-lg mx-auto">Pick a sample clip and run the analysis. The dashboard shows the prediction, confidence, and the phonetic markers that justify it.</p>
        </div>

        <div className="grid lg:grid-cols-2 gap-5">
          {/* Input panel */}
          <div className="card p-6">
            <div className="font-display text-base font-bold mb-1">Speech input</div>
            <div className="text-sm text-muted mb-5">Upload a clip or pick a sample to see how the model reasons about it.</div>

            <div className="relative mb-5">
              <button
                className="btn-ghost w-full flex justify-between items-center px-4 py-3 text-sm"
                onClick={() => setPickerOpen((o) => !o)}
                aria-expanded={pickerOpen}
              >
                <span className="flex items-center gap-2">
                  <Play size={15} className="text-fuchsia-500" />
                  {selected.label} · <span className="font-mono text-muted">{selected.speaker}</span>
                </span>
                <ChevronDown size={16} style={{ transform: pickerOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
              </button>
              {pickerOpen && (
                <div className="card absolute top-full mt-1.5 left-0 right-0 z-10 overflow-hidden">
                  {SAMPLES.map((s) => (
                    <button
                      key={s.id}
                      className="sample-row w-full text-left px-4 py-2.5 text-sm transition-colors"
                      onClick={() => reset(s.id)}
                    >
                      {s.label} <span className="font-mono text-muted">· {s.speaker}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            <Waveform bars={wave} markers={status === 'done' ? selected.markers : null} />

            <div className="flex gap-2.5 mt-5 flex-wrap">
              <button className="btn-ghost px-4 py-2.5 text-sm flex gap-2 items-center">
                <Mic size={15} /> Record
              </button>
              <button className="btn-ghost px-4 py-2.5 text-sm flex gap-2 items-center">
                <Upload size={15} /> Upload file
              </button>
              <button
                className="btn-primary px-5 py-2.5 text-sm flex gap-2 items-center ml-auto"
                onClick={analyze}
                disabled={status === 'analyzing'}
              >
                {status === 'analyzing' ? (
                  <><Loader2 size={15} style={{ animation: 'spin 0.9s linear infinite' }} /> Analyzing…</>
                ) : (
                  <><Sparkles size={15} /> Analyze speech</>
                )}
              </button>
            </div>
          </div>

          {/* Results panel */}
          <div className="card p-6 flex flex-col gap-5">
            {status !== 'done' ? (
              <EmptyState analyzing={status === 'analyzing'} />
            ) : result ? (
              <>
                <div>
                  <div className="font-mono text-xs text-muted mb-1.5">PREDICTION</div>
                  <div className="flex items-center gap-4">
                    <div className="font-display text-2xl font-bold text-fuchsia-500">{result.prediction}</div>
                    <ConfidenceGauge value={result.confidence} />
                  </div>
                </div>

                <div>
                  <div className="font-mono text-xs text-muted mb-2">ALSO CONSIDERED</div>
                  {result.alternatives.map((a: Alt) => (
                    <div key={a.lang} className="flex items-center gap-2.5 mb-1.5">
                      <span className="text-sm w-40">{a.lang}</span>
                      <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                        <div className="h-full rounded-full bg-cyan-400" style={{ width: `${a.score * 4}%` }} />
                      </div>
                      <span className="font-mono text-xs text-muted w-8 text-right">{a.score}%</span>
                    </div>
                  ))}
                </div>

                <div>
                  <div className="font-mono text-xs text-muted mb-2">WHY THIS PREDICTION</div>
                  <div className="flex flex-col gap-2">
                    {result.markers.map((m: Marker, i: number) => (
                      <div key={i} className="flex gap-2.5 items-start">
                        <span className="w-2 h-2 rounded-full mt-1.5 shrink-0" style={{ background: MARKER_COLORS[i] }} />
                        <div>
                          <div className="text-sm font-semibold">{m.label}</div>
                          <div className="text-xs text-muted">{m.note}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex items-center gap-2.5 pt-2 border-t border-white/10">
                  <MapPin size={16} className="text-cyan-400" />
                  <div>
                    <div className="text-sm font-semibold">{result.region}</div>
                    <div className="text-xs text-muted">{result.regionNote}</div>
                  </div>
                </div>
              </>
            ) : null}
          </div>
        </div>

        <div className="flex items-center justify-center gap-2 text-xs text-muted mt-6">
          <Info size={13} />
          Dashboard shown with sample predictions — the classification model and inference API are the next build phase.
        </div>
      </section>

      {/* CTA */}
      <section className="max-w-4xl mx-auto px-5 pb-20">
        <div className="card-glass p-10 text-center relative overflow-hidden">
          <div className="absolute inset-0 pointer-events-none">
            <div className="absolute -top-10 left-1/4 w-40 h-40 rounded-full bg-fuchsia-500/20 blur-2xl" />
            <div className="absolute -bottom-10 right-1/4 w-40 h-40 rounded-full bg-violet-600/20 blur-2xl" />
          </div>
          <div className="relative">
            <h2 className="font-display text-2xl sm:text-3xl font-bold">Bring AccentSense to your research</h2>
            <p className="mt-3 text-muted max-w-md mx-auto">The inference API and model weights are the next phase. Sign up to be notified when the beta opens.</p>
            <div className="mt-6 flex flex-wrap gap-3 justify-center">
              <button className="btn-primary px-6 py-3 text-sm flex items-center gap-2">
                <Sparkles size={15} /> Join the beta waitlist
              </button>
              <button className="btn-ghost px-6 py-3 text-sm flex items-center gap-2">
                <Info size={15} /> Read the whitepaper
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/5 py-8">
        <div className="max-w-6xl mx-auto px-5 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center">
              <Ear size={15} className="text-ink-900" />
            </div>
            <span className="font-display font-bold">AccentSense</span>
            <span className="font-mono text-xs text-muted ml-2">explainable accent detection</span>
          </div>
          <div className="text-xs text-muted font-mono">© 2026 AccentSense · Research preview</div>
        </div>
      </footer>
    </div>
  );
}

function EmptyState({ analyzing }: { analyzing: boolean }) {
  return (
    <div className="flex flex-col items-center justify-center flex-1 min-h-[260px] text-center text-muted">
      {analyzing ? (
        <>
          <Loader2 size={28} className="text-fuchsia-500 mb-4" style={{ animation: 'spin 0.9s linear infinite' }} />
          <div className="text-sm">Running feature extraction + classification…</div>
        </>
      ) : (
        <>
          <Sparkles size={28} className="text-violet-500 mb-4 opacity-70" />
          <div className="text-sm">Results will appear here after you analyze a clip.</div>
        </>
      )}
    </div>
  );
}

function ConfidenceGauge({ value }: { value: number }) {
  const r = 20;
  const c = 2 * Math.PI * r;
  const offset = c - (value / 100) * c;
  return (
    <div className="relative w-[52px] h-[52px]">
      <svg width="52" height="52" viewBox="0 0 52 52">
        <circle cx="26" cy="26" r={r} fill="none" stroke="rgba(241,240,251,0.12)" strokeWidth="5" />
        <circle
          cx="26" cy="26" r={r} fill="none" stroke="#8B5CF6" strokeWidth="5"
          strokeDasharray={c} strokeDashoffset={offset} strokeLinecap="round"
          transform="rotate(-90 26 26)"
        />
      </svg>
      <div className="font-mono absolute inset-0 flex items-center justify-center text-xs font-semibold">
        {value}%
      </div>
    </div>
  );
}

function Waveform({ bars, markers }: { bars: number[]; markers: readonly Marker[] | null }) {
  const width = 600;
  const height = 100;
  const barW = width / bars.length;

  return (
    <div className="card px-3.5 py-3 pb-2.5" style={{ background: '#0F1338' }}>
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" height="90" preserveAspectRatio="none">
        {markers && markers.map((m, i) => (
          <rect
            key={i}
            className="marker-band"
            x={(m.start / 90) * width} y={0}
            width={((m.end - m.start) / 90) * width} height={height}
            fill={MARKER_COLORS[i]} opacity={0.14}
          />
        ))}
        {bars.map((v, i) => (
          <rect
            key={i}
            x={i * barW + barW * 0.25}
            y={(height - v * height) / 2}
            width={barW * 0.5}
            height={v * height}
            rx={1.5}
            fill={markers ? '#E879F9' : '#8B5CF6'}
            opacity={markers ? 0.9 : 0.75}
          />
        ))}
      </svg>
      {markers && (
        <div className="flex gap-3.5 flex-wrap mt-1 pt-1.5">
          {markers.map((m, i) => (
            <div key={i} className="flex items-center gap-1.5">
              <span className="w-[7px] h-[7px] rounded-full" style={{ background: MARKER_COLORS[i] }} />
              <span className="font-mono text-[10px] text-muted">{m.label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
