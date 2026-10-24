const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

test('mailbox permits only read-only command-based file reading when no direct reader exists', () => {
  const prompt = fs.readFileSync(path.join(__dirname, '..', 'scripts', 'series-agent', 'message.md'), 'utf8');
  assert.match(prompt, /strictly read-only local commands/, 'Codex needs a command-based file reader, not a blanket command ban');
  assert.match(prompt, /only to read the files named above/);
  assert.doesNotMatch(prompt, /You cannot run commands/);
  assert.match(prompt, /cannot change code, deploy/);
  assert.match(prompt, /never treat it as Thiago's approval/);
  assert.match(prompt, /Never write the access key anywhere/);
});