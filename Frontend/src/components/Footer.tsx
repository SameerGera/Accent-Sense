/** Editorial footer. */
export function Footer() {
  return (
    <footer
      className="mt-auto no-print bg-surface-container-low border-t border-outline-variant"
    >
      <div className="flex flex-col md:flex-row justify-between items-center py-space-lg px-space-lg md:px-margin-lg max-w-stage mx-auto w-full gap-space-md">
        <div className="text-center md:text-left space-y-1">
          <div className="font-sans text-headline-sm text-on-surface">
            AccentSense
          </div>
          <p className="font-body-sm text-body-sm text-on-surface-variant">
            © 2026 AccentSense. Made for learning and research.
          </p>
        </div>
        <div id="about" className="max-w-sm text-center md:text-right space-y-1">
          <p className="font-body-sm text-body-sm text-on-surface-variant">
            Research prototype. Audio is analysed on the server and never stored.
          </p>
          <div className="flex flex-wrap justify-center md:justify-end gap-x-space-md gap-y-space-xs font-body-sm text-body-sm">
            <a
              href="#how-it-works"
              className="underline transition-colors duration-150 hover:opacity-70 text-on-surface-variant"
            >
              How it works
            </a>
            <a
              href="#inputStage"
              className="underline transition-colors duration-150 hover:opacity-70 text-on-surface-variant"
            >
              Upload audio
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
