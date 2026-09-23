import { ThemeToggle } from './ThemeToggle';

export interface HeaderProps {
  onStartOver: () => void;
  canStartOver: boolean;
}

/** Sticky 60px top nav — matches Stitch TopNavBar blueprint. */
export function Header({ onStartOver, canStartOver }: HeaderProps) {
  return (
    <header
      className="sticky top-0 z-50 no-print header-shell bg-surface border-b border-outline-variant"
    >
      <div className="flex justify-between items-center h-[60px] px-space-lg md:px-margin-lg max-w-stage mx-auto w-full gap-space-sm">
        {/* Brand */}
        <a
          href="#top"
          className="font-headline text-headline-md tracking-tight shrink-0 transition-opacity hover:opacity-70 text-on-surface"
        >
          AccentSense
        </a>

        {/* Nav links — always available; denser on mobile */}
        <nav
          className="flex items-center gap-space-sm md:gap-space-lg font-body-md text-body-sm md:text-body-md min-w-0 overflow-hidden"
          aria-label="Primary"
        >
          <a href="#how-it-works" className="nav-link nav-link-active" aria-current="page">
            How it works
          </a>
          <a href="#about" className="nav-link">
            About
          </a>
        </nav>

        {/* Trailing controls */}
        <div className="flex items-center gap-space-sm shrink-0">
          <ThemeToggle />
          <button
            type="button"
            onClick={onStartOver}
            disabled={!canStartOver}
            className="btn-secondary !h-[30px] !px-2 md:!px-3 !text-label-code disabled:opacity-40 disabled:cursor-not-allowed whitespace-nowrap"
            title={canStartOver ? 'Clear session and start over' : 'Nothing to reset yet'}
          >
            <span className="hidden sm:inline">Start over</span>
            <span className="sm:hidden material-symbols-outlined !text-[16px]" aria-hidden="true">
              refresh
            </span>
          </button>
        </div>
      </div>
    </header>
  );
}
