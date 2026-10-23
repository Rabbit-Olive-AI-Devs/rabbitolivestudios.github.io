/**
 * Where this Mac keeps Dropbox and the Photos library, so the same jobs run on the Mac mini or the
 * MacBook (DECISIONS #99). ~/.config/eink/paths.json, every key optional:
 *
 *   {"dropbox": "/Users/quark/Dropbox",
 *    "photosLibrary": "/Users/quark/Pictures/Photos Library.photoslibrary",
 *    "familyTmp": "/Users/quark/Pictures/.eink-family-tmp"}
 *
 * Without the file: the Mac mini's external SSD. scripts/storybook-epub.py reads the same file.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const SSD = "/Volumes/SamsungSSD";

export function einkPaths(home = os.homedir()) {
  let file = {};
  try { file = JSON.parse(fs.readFileSync(path.join(home, ".config", "eink", "paths.json"), "utf8")); } catch { /* the mini's defaults */ }
  return {
    dropbox: file.dropbox ?? `${SSD}/.CloudStorage/Data/Dropbox`,
    photosLibrary: file.photosLibrary ?? `${SSD}/Pictures/Photos Library.photoslibrary`,
    familyTmp: file.familyTmp ?? `${SSD}/Pictures/.eink-family-tmp`,
  };
}
