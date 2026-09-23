import { motion } from 'framer-motion';
import { heroContainer, heroLine } from '../lib/motion';

/** Editorial introduction — kicker, display headline, subhead (staggered entrance). */
export function EditorialIntro() {
  return (
    <motion.section
      className="space-y-space-md"
      variants={heroContainer}
      initial="hidden"
      animate="show"
    >
      <motion.div className="flex items-center gap-space-sm" variants={heroLine}>
        <span className="kicker text-primary-container">
          Speech Analysis
        </span>
      </motion.div>
      <motion.h1
        className="font-headline text-display-lg-mobile md:text-display-lg leading-tight tracking-tight text-on-surface"
        variants={heroLine}
      >
        Find out what your English accent reveals about where you&rsquo;re from.
      </motion.h1>
      <motion.p
        className="font-body-lg text-body-lg leading-relaxed text-on-surface-variant"
        variants={heroLine}
      >
        Upload a short voice recording. We analyse your rhythm, vowels, and
        consonants to estimate which UK regional accent you speak with — and
        show which parts of the audio drove the result.
      </motion.p>
    </motion.section>
  );
}
