import { motion } from 'framer-motion';
import { fadeRise } from '../lib/motion';

/** Static "How it works" explainer — anchor target for the nav link. */
export function HowItWorks() {
  return (
    <motion.section
      id="how-it-works"
      className="grid grid-cols-1 md:grid-cols-2 gap-space-lg pt-space-sm border-t border-outline-variant"
      variants={fadeRise}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, margin: '-40px' }}
    >
      <div className="space-y-space-xs">
        <h5 className="font-sans text-headline-sm text-on-surface">
          Background
        </h5>
        <p className="font-body-md text-body-md leading-relaxed text-on-surface-variant">
          Regional accents follow measurable patterns in rhythm, vowels, and
          consonants across England, Scotland, Wales, and Ireland.
        </p>
      </div>
      <div className="space-y-space-xs">
        <h5 className="font-sans text-headline-sm text-on-surface">
          How this works
        </h5>
        <p className="font-body-md text-body-md leading-relaxed text-on-surface-variant">
          A YAMNet frame encoder embeds your audio in ~0.48-second frames;
          an accent classifier scores each frame across seven classes
          (Irish, Midlands, Northern, Scottish, Southern, Welsh, and
          “Not a speech”), averaged into one result. The timeline shows
          where the model's score was highest — model scores are not
          calibrated confidences. Audio is never stored.
        </p>
      </div>
    </motion.section>
  );
}
