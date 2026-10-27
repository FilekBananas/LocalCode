**Document basics.** Every page: `<!doctype html>`, `<html lang="…">`, `<meta charset="utf-8">`, `<meta name="viewport" content="width=device-width, initial-scale=1">`, a descriptive `<title>`. Link CSS in `<head>`, defer scripts.

**Semantic HTML.** Use elements for meaning, not just layout: `<header>`, `<nav>`, `<main>` (one per page), `<section>`, `<article>`, `<aside>`, `<footer>`, `<figure>`/`<figcaption>`. Headings `<h1>`–`<h6>` in order without skipping. `<button>` for in-page actions, `<a href>` for navigation (don't fake either with a `<div>` + JS). Forms: every control has a `<label for>` (or wraps it), group with `<fieldset>`/`<legend>`, correct `type`/`name`/`required`/`autocomplete`.

**Accessibility (a11y).** Images need meaningful `alt` (empty `alt=""` for decorative). Maintain color contrast (WCAG AA: 4.5:1 text). Everything operable by keyboard; visible focus styles (don't remove `:focus` outlines without a replacement). Use native semantics first; add ARIA (`role`, `aria-label`, `aria-expanded`…) only to fill gaps, never to override correct semantics. Respect `prefers-reduced-motion`.

**CSS architecture.** External stylesheet(s). Target with classes; keep specificity low and flat — avoid deep descendant chains, ID selectors for styling, and `!important`. Consider a convention (BEM, utility classes) consistent with the repo. CSS custom properties (`--color-bg`) for theme tokens; `@media (prefers-color-scheme: dark)` for dark mode.

**Layout & responsiveness.** Mobile-first: base styles for small screens, `@media (min-width: …)` to enhance. Use Flexbox for 1-D layouts, Grid for 2-D. Relative units (`rem` for type/spacing, `%`/`fr`/`minmax` for tracks, `clamp()` for fluid sizing); avoid fixed pixel widths that overflow. Box model: set `box-sizing: border-box`. Test at narrow and wide viewports and for text zoom.

**Performance.** Minimize render-blocking CSS, use `loading="lazy"` on offscreen images, set width/height (or `aspect-ratio`) to avoid layout shift, prefer modern image formats.

**Validation & testing.** Validate HTML (W3C validator), check contrast and keyboard navigation, and verify across the layout breakpoints. Keep JS progressive — the page should be usable before scripts run where feasible.

**Pitfalls.** `<div>`-soup instead of semantic/interactive elements; missing `alt`/labels; removing focus outlines; non-responsive fixed widths and horizontal scroll on mobile; over-specific CSS and `!important` wars; inline styles; blocking the whole page on a slow script.
