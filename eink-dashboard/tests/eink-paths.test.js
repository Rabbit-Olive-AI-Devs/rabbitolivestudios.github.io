const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const paths = () => import(path.join(__dirname, "..", "scripts", "eink-paths.mjs"));

test("einkPaths: the Mac mini's SSD without a file; any key the file sets wins", async () => {
  const { einkPaths } = await paths();
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "paths-home-"));
  assert.equal(einkPaths(home).dropbox, "/Volumes/SamsungSSD/.CloudStorage/Data/Dropbox");
  fs.mkdirSync(path.join(home, ".config", "eink"), { recursive: true });
  fs.writeFileSync(path.join(home, ".config", "eink", "paths.json"), JSON.stringify({ dropbox: "/Users/quark/Dropbox" }));
  const p = einkPaths(home);
  assert.equal(p.dropbox, "/Users/quark/Dropbox");
  assert.equal(p.photosLibrary, "/Volumes/SamsungSSD/Pictures/Photos Library.photoslibrary");
});
