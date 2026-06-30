/**
 * Per-display stylesheets for the World Cup pages — DELIBERATELY SEPARATE so a change to one
 * display never affects the other (DECISIONS #48). Shared structure/logic lives in worldcup-ui.ts;
 * only the CSS diverges. COLOR_STYLE = pristine pre-session E1002 (with flags); the B&W page
 * composes MONO_STYLE_BASE (crisp 3.15.0 layout) with its inlined Inter font in pages/worldcup.ts.
 */

/** E1002 color page stylesheet (Spectra-6; pristine pre-session). */
export const COLOR_STYLE = `  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { width: 800px; height: 480px; overflow: hidden; display: flex; flex-direction: column;
    background: #fff; color: #000; font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; padding: 14px 22px; }
  .wc-header { display: flex; justify-content: space-between; align-items: baseline; border-bottom: 3px solid #000; padding-bottom: 6px; margin-bottom: 8px; }
  .wc-title { font-size: 24px; font-weight: 800; letter-spacing: 1px; }
  .wc-sub { font-size: 17px; font-weight: 600; }
  .wc-panel-label { font-size: 15px; font-weight: 800; letter-spacing: 1.5px; margin-bottom: 4px; }
  .wc-split { display: flex; gap: 24px; flex: 1; min-height: 0; }
  .wc-col { flex: 1; display: flex; flex-direction: column; min-height: 0; }
  .wc-rowlist { flex: 1; display: flex; flex-direction: column; justify-content: space-evenly; min-height: 0; overflow: hidden; }
  .wc-row { font-size: 21px; font-weight: 700; display: flex; flex-wrap: nowrap; align-items: center; gap: 7px; overflow: hidden; }
  .wc-cell { display: inline-block; min-width: 62px; flex: 0 0 auto; font-weight: 800; }
  .wc-v { font-size: 15px; font-weight: 500; flex: 0 0 auto; }
  .wc-team { display: inline-flex; align-items: center; white-space: nowrap; flex: 0 1 auto; min-width: 0; }
  .wc-flag { height: 15px; width: auto; vertical-align: middle; margin-right: 7px; border: 1px solid #000; }
  .wc-flag-big { height: 150px; width: auto; border: 2px solid #000; image-rendering: pixelated; margin-bottom: 6px; }
  .wc-bottom { flex: 1.15; min-height: 0; overflow: hidden; border-top: 3px solid #000; padding-top: 6px; display: flex; flex-direction: column; }
  .wc-group { flex: 1; display: flex; flex-direction: column; min-height: 0; }
  .wc-group-name { font-size: 18px; font-weight: 800; letter-spacing: 1px; margin-bottom: 4px; }
  .wc-table { width: 100%; height: 100%; border-collapse: collapse; font-size: 22px; }
  .wc-table th { font-size: 14px; font-weight: 700; text-align: center; padding: 1px 6px; }
  .wc-table td { text-align: center; padding: 2px 6px; font-weight: 700; }
  .wc-table td:nth-child(2) { text-align: left; font-weight: 700; }
  .wc-pos { color: #000; } .wc-pts { font-weight: 800; }
  /* Round-of-32 knockout bracket (color: with flags) */
  .wc-kbracket { flex: 1; min-height: 0; display: flex; gap: 3px; align-items: stretch; overflow: hidden; }
  .wc-kside { flex: 1; display: flex; gap: 3px; min-width: 0; }
  .wc-kcol { display: flex; flex-direction: column; justify-content: space-around; min-width: 0; }
  .wc-kr32 { flex: 0 0 146px; }
  .wc-kside .wc-kcol:not(.wc-kr32) { flex: 1; }
  .wc-kcenter { flex: 0 0 50px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 5px; }
  .wc-kcenter-label { font-size: 10px; font-weight: 800; letter-spacing: 1px; }
  .wc-ktie { border: 1.5px solid #000; border-radius: 4px; padding: 1px 6px; display: flex; flex-direction: column; justify-content: center; overflow: hidden; }
  .wc-kteam { font-size: 14px; font-weight: 600; line-height: 16px; display: flex; align-items: center; gap: 6px; white-space: nowrap; overflow: hidden; }
  /* name only grows (pushing the score to the right edge) when a score is present; otherwise it
     collapses so an advancing flag/code stays centered by the box's justify-content. */
  .wc-kname { overflow: hidden; text-overflow: ellipsis; flex: 0 1 auto; min-width: 0; }
  .wc-kteam-scored .wc-kname { flex: 1 1 auto; }
  .wc-kscore { font-weight: 800; flex: 0 0 auto; padding-left: 4px; }
  .wc-kwhen { font-size: 9px; font-weight: 600; line-height: 11px; white-space: nowrap; overflow: hidden; opacity: 0.85; }
  .wc-ktie .wc-flag { height: 11px; width: auto; border: 1px solid #000; margin: 0; flex: 0 0 auto; }
  .wc-kempty { flex: 0 0 auto; min-height: 46px; align-items: center; justify-content: center; }
  .wc-ktbd { font-size: 11px; font-weight: 700; line-height: 13px; text-align: center; white-space: nowrap; overflow: hidden; }
  /* Inner rounds (everything but the wide R32 columns): bigger centered flags for an advancing team
     (the date/time sits on one line below to leave them the room); the date/time is centered. */
  .wc-kcol:not(.wc-kr32) .wc-ktie, .wc-kfinal { min-height: 58px; }
  .wc-kcol:not(.wc-kr32) .wc-kteam, .wc-kfinal .wc-kteam { justify-content: center; line-height: 24px; }
  .wc-kcol:not(.wc-kr32) .wc-kteam, .wc-kfinal .wc-kteam { gap: 4px; }
  .wc-kcol:not(.wc-kr32) .wc-flag, .wc-kfinal .wc-flag { height: 22px; }
  /* a finished tie shows a per-team score next to the flag — keep that flag smaller so the score fits. */
  .wc-kcol:not(.wc-kr32) .wc-kteam-scored .wc-flag, .wc-kfinal .wc-kteam-scored .wc-flag { height: 16px; }
  .wc-kcol:not(.wc-kr32) .wc-kscore, .wc-kfinal .wc-kscore { font-size: 11px; padding-left: 2px; }
  .wc-kcol:not(.wc-kr32) .wc-kwhen, .wc-kfinal .wc-kwhen { text-align: center; font-size: 8px; }
  .wc-klive { border-width: 2.5px; }
  .wc-champion { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 14px; }
  .wc-champ-label { font-size: 22px; font-weight: 700; letter-spacing: 4px; }
  .wc-champ-team { font-size: 104px; font-weight: 800; line-height: 1; }
  .wc-champ-final { font-size: 30px; font-weight: 700; }
  .wc-empty { font-size: 18px; color: #000; padding: 6px 0; }`;

/** E1001 B&W page stylesheet base (crispness layout; weight 500, no clip, content-sized). */
export const MONO_STYLE_BASE = `  /* E-ink crispness: disable font anti-aliasing so the mono panel gets pure 1-bit glyph
     edges instead of grayscale-smoothed ones it would quantize into smudge/serration.
     See DECISIONS #48 (refs: SumatraPDF#599, MobileRead e-ink font threads). */
  * { margin: 0; padding: 0; box-sizing: border-box;
    -webkit-font-smoothing: none; font-smooth: never; text-rendering: optimizeSpeed; }
  body { width: 800px; height: 480px; overflow: hidden; display: flex; flex-direction: column;
    background: #fff; color: #000; font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; padding: 14px 22px; }
  .wc-header { display: flex; justify-content: space-between; align-items: baseline; border-bottom: 3px solid #000; padding-bottom: 6px; margin-bottom: 8px; }
  .wc-title { font-size: 24px; font-weight: 700; letter-spacing: 1px; }
  .wc-sub { font-size: 17px; font-weight: 500; }
  .wc-panel-label { font-size: 15px; font-weight: 700; letter-spacing: 1.5px; margin-bottom: 6px; }
  .wc-split { display: flex; gap: 24px; flex: 1; min-height: 0; margin-bottom: 14px; }
  .wc-col { flex: 1; display: flex; flex-direction: column; min-height: 0; }
  .wc-rowlist { flex: 1; display: flex; flex-direction: column; justify-content: flex-start; gap: 6px; min-height: 0; overflow: hidden; }
  /* flex-shrink:0 + overflow:visible: rows keep their full line box so flexbox can't crush
     6 rows into 14px and clip the text (descenders). See DECISIONS #46. */
  .wc-row { font-size: 21px; font-weight: 500; line-height: 25px; flex-shrink: 0; display: flex; flex-wrap: nowrap; align-items: center; gap: 7px; min-width: 0; overflow: visible; }
  .wc-cell { display: inline-block; min-width: 62px; flex: 0 0 auto; font-weight: 700; }
  .wc-v { font-size: 15px; font-weight: 500; flex: 0 0 auto; }
  .wc-team { display: inline-flex; align-items: center; white-space: nowrap; flex: 0 1 auto; min-width: 0; }
  .wc-flag { height: 15px; width: auto; vertical-align: middle; margin-right: 7px; border: 1px solid #000; }
  .wc-flag-big { height: 150px; width: auto; border: 2px solid #000; image-rendering: pixelated; margin-bottom: 6px; }
  .wc-bottom { flex: 0 0 auto; min-height: 0; overflow: hidden; border-top: 3px solid #000; padding-top: 6px; display: flex; flex-direction: column; }
  .wc-group { flex: 1; display: flex; flex-direction: column; min-height: 0; }
  .wc-group-name { font-size: 18px; font-weight: 700; letter-spacing: 1px; margin-bottom: 3px; }
  .wc-table { width: 100%; border-collapse: collapse; font-size: 22px; }
  .wc-table th { font-size: 14px; font-weight: 500; text-align: center; padding: 1px 6px; }
  .wc-table td { text-align: center; padding: 3px 6px; font-weight: 500; line-height: 24px; }
  .wc-table td:nth-child(2) { text-align: left; font-weight: 500; }
  .wc-table td.wc-pos { font-weight: 500; } .wc-table td.wc-pts { font-weight: 700; }
  /* Round-of-32 knockout bracket (mono: names only, no flags) */
  .wc-kbracket { flex: 1; min-height: 0; display: flex; gap: 3px; align-items: stretch; overflow: hidden; }
  .wc-kside { flex: 1; display: flex; gap: 3px; min-width: 0; }
  .wc-kcol { display: flex; flex-direction: column; justify-content: space-around; min-width: 0; }
  .wc-kr32 { flex: 0 0 128px; }
  .wc-kside .wc-kcol:not(.wc-kr32) { flex: 1; }
  .wc-kcenter { flex: 0 0 50px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 5px; }
  .wc-kcenter-label { font-size: 10px; font-weight: 700; letter-spacing: 1px; }
  .wc-ktie { border: 1.5px solid #000; border-radius: 4px; padding: 1px 7px; display: flex; flex-direction: column; justify-content: center; overflow: hidden; }
  .wc-kteam { font-size: 15px; font-weight: 600; line-height: 17px; display: flex; align-items: center; white-space: nowrap; overflow: hidden; }
  /* name only grows (pushing the score right) when a score is present; otherwise it collapses so
     an advancing 3-letter code stays centered by the box's justify-content. */
  .wc-kname { overflow: hidden; text-overflow: ellipsis; flex: 0 1 auto; min-width: 0; }
  .wc-kteam-scored .wc-kname { flex: 1 1 auto; }
  .wc-kscore { font-weight: 800; flex: 0 0 auto; padding-left: 4px; }
  .wc-kwhen { font-size: 10px; font-weight: 600; line-height: 12px; white-space: nowrap; overflow: hidden; }
  .wc-kempty { flex: 0 0 auto; min-height: 44px; align-items: center; justify-content: center; }
  .wc-ktbd { font-size: 12px; font-weight: 700; line-height: 14px; text-align: center; white-space: nowrap; overflow: hidden; }
  /* Inner rounds: centered 3-letter code for an advancing team, or code + per-team score on a
     finished tie. Slightly smaller, with a smaller score, so "GER 1(4)" fits without truncating
     the code; the date/time is one line, centered. */
  .wc-kcol:not(.wc-kr32) .wc-ktie, .wc-kfinal { min-height: 52px; }
  .wc-kcol:not(.wc-kr32) .wc-kteam, .wc-kfinal .wc-kteam { justify-content: center; font-size: 13px; line-height: 16px; }
  .wc-kcol:not(.wc-kr32) .wc-kscore, .wc-kfinal .wc-kscore { font-size: 11px; padding-left: 2px; }
  .wc-kcol:not(.wc-kr32) .wc-kwhen, .wc-kfinal .wc-kwhen { text-align: center; font-size: 9px; }
  .wc-champion { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 14px; }
  .wc-champ-label { font-size: 22px; font-weight: 700; letter-spacing: 4px; }
  .wc-champ-team { font-size: 104px; font-weight: 700; line-height: 1; }
  .wc-champ-final { font-size: 30px; font-weight: 700; }
  .wc-empty { font-size: 18px; color: #000; padding: 6px 0; }`;
