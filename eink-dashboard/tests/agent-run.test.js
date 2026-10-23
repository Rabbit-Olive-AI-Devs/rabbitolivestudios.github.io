const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

// A stand-in agent command: records its arguments and stdin, and answers like the real one
// (on stdout, or in the file after -o as Codex does). No test ever starts a real agent.
function fakeAgent() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "fake-agent-"));
  const bin = path.join(dir, "agent");
  fs.writeFileSync(bin, `#!/bin/sh
printf '%s\\n' "$@" > "${dir}/args"
cat > "${dir}/stdin"
out=""; prev=""
for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done
if [ -n "$out" ]; then printf 'VERDICT: PASS' > "$out"; else printf 'VERDICT: PASS'; fi
`, { mode: 0o755 });
  return { bin, args: () => fs.readFileSync(path.join(dir, "args"), "utf8").trim().split("\n"), stdin: () => fs.readFileSync(path.join(dir, "stdin"), "utf8") };
}

/** The runner loaded with a temporary home holding this agent.json (null: no file). */
async function runnerWith(config) {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "agent-home-"));
  if (config) {
    fs.mkdirSync(path.join(home, ".config", "eink"), { recursive: true });
    fs.writeFileSync(path.join(home, ".config", "eink", "agent.json"), JSON.stringify(config));
  }
  const saved = process.env.HOME;
  process.env.HOME = home;
  try {
    return await import(`${path.join(__dirname, "..", "scripts", "agent-run.mjs")}?home=${encodeURIComponent(home)}`);
  } finally { process.env.HOME = saved; }
}

const after = (a, flag, n = 1) => a.slice(a.indexOf(flag), a.indexOf(flag) + 1 + n);

test("agentConfig: Claude when nothing is set; EINK_AGENT overrides the file", async () => {
  assert.equal((await runnerWith(null)).agentConfig().agent, "claude");
  const m = await runnerWith({ agent: "codex", model: "m1" });
  assert.equal(m.agentConfig().agent, "codex");
  process.env.EINK_AGENT = "claude";
  try { assert.equal(m.agentConfig().agent, "claude"); } finally { delete process.env.EINK_AGENT; }
});

test("runAgent: Claude keeps its exact tool rules and directories", async () => {
  const fake = fakeAgent();
  const { runAgent } = await runnerWith({ agent: "claude", command: fake.bin });
  const r = runAgent({ prompt: "P", cwd: os.tmpdir(), read: ["/r"], write: ["/w"], claudeTools: ["Read", "Write(//w/**)"] });
  assert.equal(r.stdout, "VERDICT: PASS");
  const a = fake.args();
  assert.deepEqual(after(a, "-p"), ["-p", "P"]);
  assert.deepEqual(after(a, "--model"), ["--model", "opus"]);
  assert.deepEqual(after(a, "--allowedTools"), ["--allowedTools", "Read Write(//w/**)"]);
  assert.deepEqual(a.slice(-4), ["--add-dir", "/r", "--add-dir", "/w"]);
});

test("runAgent: Codex gets the prompt on stdin, a sandbox, writable dirs, images, and its answer read back", async () => {
  const fake = fakeAgent();
  const { runAgent } = await runnerWith({ agent: "codex", model: "m1", command: fake.bin });
  const ro = runAgent({ prompt: "look", cwd: "/repo", images: ["/p.png"], claudeTools: ["Read"] });
  assert.equal(ro.stdout, "VERDICT: PASS");
  assert.equal(fake.stdin(), "look");
  let a = fake.args();
  assert.deepEqual(after(a, "-s", 3), ["-s", "read-only", "-C", "/repo"]);
  assert.ok(a.includes("--ignore-user-config"), "never the user's Codex config: its memories would keep the story");
  assert.deepEqual(after(a, "-m"), ["-m", "m1"]);
  assert.deepEqual(after(a, "-i"), ["-i", "/p.png"]);

  runAgent({ prompt: "write", cwd: "/repo", write: ["/work", "/outbox"], web: true, claudeTools: [] });
  a = fake.args();
  assert.deepEqual(after(a, "-s", 3), ["-s", "workspace-write", "-C", "/work"], "a writing run works in its first writable dir, never the repo");
  assert.deepEqual(after(a, "--add-dir"), ["--add-dir", "/outbox"]);
  assert.ok(a.includes("sandbox_workspace_write.network_access=true"));
});

test("runAgent: an unknown agent fails without running anything", async () => {
  const { runAgent } = await runnerWith({ agent: "nobody" });
  const r = runAgent({ prompt: "P", cwd: os.tmpdir(), claudeTools: [] });
  assert.equal(r.status, 2);
  assert.match(r.stderr, /unknown agent "nobody"/);
});
