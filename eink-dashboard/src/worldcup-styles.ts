/**
 * Per-display stylesheets for the World Cup pages — DELIBERATELY SEPARATE so a change to one
 * display never affects the other (DECISIONS #48). Shared structure/logic lives in worldcup-ui.ts;
 * only the CSS diverges. COLOR_STYLE is the pristine pre-session E1002 look (with flags); the B&W
 * page composes MONO_STYLE_BASE with its inlined Inter font in pages/worldcup.ts.
 */

/** E1002 color page stylesheet (Spectra-6; pristine pre-session). */
export const COLOR_STYLE = ` }
  * { margin: 0; padding: 0; box-sizing: border-box; }
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
  .wc-r32 { columns: 2; font-size: 21px; flex: 1; }
  .wc-bracket { flex: 1; display: flex; gap: 5px; min-height: 0; overflow: hidden; align-items: stretch; }
  .wc-bcol { flex: 1; display: flex; flex-direction: column; justify-content: space-around; }
  .wc-bcol-label { font-size: 13px; font-weight: 800; text-align: center; letter-spacing: 1px; margin-bottom: 4px; }
  .wc-tie { border: 2px solid #000; border-radius: 4px; padding: 4px 4px; margin: 4px 0; text-align: center; }
  .wc-bteam { font-size: 19px; font-weight: 700; display: flex; align-items: center; justify-content: center; gap: 4px; }
  .wc-bteam .wc-flag { height: 13px; margin-right: 0; }
  .wc-bscore { font-size: 15px; font-weight: 800; }
  .wc-third { font-size: 17px; font-weight: 700; text-align: center; margin-top: 6px; }
  .wc-champion { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 14px; }
  .wc-champ-label { font-size: 22px; font-weight: 700; letter-spacing: 4px; }
  .wc-champ-team { font-size: 104px; font-weight: 800; line-height: 1; }
  .wc-champ-final { font-size: 30px; font-weight: 700; }
  .wc-empty { font-size: 18px; color: #000; padding: 6px 0; }
  `;

/** E1001 B&W page stylesheet base (crispness layout; weight 500, no clip, content-sized). */
export const MONO_STYLE_BASE = ` }
  /* E-ink crispness: disable font anti-aliasing so the mono panel gets pure 1-bit glyph
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
  .wc-r32 { columns: 2; font-size: 21px; flex: 1; }
  .wc-bracket { flex: 1; display: flex; gap: 5px; min-height: 0; overflow: hidden; align-items: stretch; }
  .wc-bcol { flex: 1; display: flex; flex-direction: column; justify-content: flex-start; gap: 12px; }
  .wc-bcol-label { font-size: 13px; font-weight: 700; text-align: center; letter-spacing: 1px; margin-bottom: 6px; }
  .wc-tie { border: 2px solid #000; border-radius: 4px; padding: 5px 4px; text-align: center; }
  .wc-bteam { font-size: 19px; font-weight: 500; line-height: 24px; display: flex; align-items: center; justify-content: center; gap: 4px; }
  .wc-bteam .wc-flag { height: 13px; margin-right: 0; }
  .wc-bscore { font-size: 15px; font-weight: 700; }
  .wc-third { font-size: 17px; font-weight: 500; text-align: center; margin-top: 6px; }
  .wc-champion { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 14px; }
  .wc-champ-label { font-size: 22px; font-weight: 700; letter-spacing: 4px; }
  .wc-champ-team { font-size: 104px; font-weight: 700; line-height: 1; }
  .wc-champ-final { font-size: 30px; font-weight: 700; }
  .wc-empty { font-size: 18px; color: #000; padding: 6px 0; }
  `;
