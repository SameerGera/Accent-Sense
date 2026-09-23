import type { Variants } from 'framer-motion';

/** Shared easing — soft editorial settle */
export const easeOut = [0.22, 1, 0.36, 1] as const;
export const easeInOut = [0.65, 0, 0.35, 1] as const;

/** Stagger children container */
export const stagger = (delay = 0, each = 0.08): Variants => ({
  hidden: {},
  show: {
    transition: { staggerChildren: each, delayChildren: delay },
  },
});

/** Fade + rise (primary section entrance) */
export const fadeRise: Variants = {
  hidden: { opacity: 0, y: 18 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.55, ease: easeOut },
  },
  exit: {
    opacity: 0,
    y: -10,
    transition: { duration: 0.28, ease: easeInOut },
  },
};

/** Soft fade only (subtler blocks) */
export const fade: Variants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { duration: 0.4, ease: easeOut } },
  exit: { opacity: 0, transition: { duration: 0.22 } },
};

/** Scale-in from center (drop zone / cards) */
export const scaleIn: Variants = {
  hidden: { opacity: 0, scale: 0.97 },
  show: {
    opacity: 1,
    scale: 1,
    transition: { duration: 0.45, ease: easeOut },
  },
  exit: {
    opacity: 0,
    scale: 0.98,
    transition: { duration: 0.2 },
  },
};

/** Slide from left (progress, horizontal reveals) */
export const slideLeft: Variants = {
  hidden: { opacity: 0, x: -16 },
  show: {
    opacity: 1,
    x: 0,
    transition: { duration: 0.5, ease: easeOut },
  },
  exit: { opacity: 0, x: -8, transition: { duration: 0.2 } },
};

/** Score bar grow (width driven by style, this animates opacity/transform) */
export const barReveal: Variants = {
  hidden: { opacity: 0, scaleX: 0 },
  show: (i: number) => ({
    opacity: 1,
    scaleX: 1,
    transition: {
      duration: 0.65,
      ease: easeOut,
      delay: 0.15 + i * 0.05,
    },
  }),
};

/** List item stagger for numbered rows / region cards */
export const listItem: Variants = {
  hidden: { opacity: 0, y: 10 },
  show: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.4, ease: easeOut, delay: i * 0.06 },
  }),
};

/** Layout transition for stage swaps (upload ↔ player) */
export const layoutSpring = {
  type: 'spring' as const,
  stiffness: 320,
  damping: 32,
  mass: 0.8,
};

/** Hero entrance on first paint */
export const heroContainer: Variants = {
  hidden: {},
  show: {
    transition: { staggerChildren: 0.1, delayChildren: 0.05 },
  },
};

export const heroLine: Variants = {
  hidden: { opacity: 0, y: 22 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.65, ease: easeOut },
  },
};
