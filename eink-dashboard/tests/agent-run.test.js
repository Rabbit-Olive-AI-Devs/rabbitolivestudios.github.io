const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

// SYNTHETIC command plumbing only: never native agent/model acceptance.
// Guard every spawn, including broken default selection, against starting a real CLI.
const childProcess = require("node:child_process");
const { syncBuiltinESMExports } = require("node:module");
const nativeSpawn = childProcess.spawnSync;
const syntheticCommands = new Set();
const calls = [];
childProcess.spawnSync = (command, args, options) => {
  calls.push({ command, args });
  if (!syntheticCommands.has(command)) {
    return { status: 97, signal: null, stdout: "", stderr: "SYNTHETIC: native execution blocked" };
  }
  return nativeSpawn(command, args, options);
};
syncBuiltinESMExports();

function fakeAgent(status = 0) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "fake-agent-"));
  const bin = path.join(dir, "agent");
  fs.writeFileSync(bin, `#!/bin/sh
printf '%s\\n' "$@" > "${dir}/args"
cat > "${dir}/stdin"
out=""; prev=""
for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done
if [ -n "$out" ]; then printf 'VERDICT: PASS' > "$out"; else printf 'VERDICT: PASS'; fi
printf 'SYNTHETIC stderr' >&2
exit ${status}
`, { mode: 0o755 });
  syntheticCommands.add(bin);
  return { bin, args: () => fs.readFileSync(path.join(dir, "args"), "utf8").trim().split("\n"), stdin: () => fs.readFileSync(path.join(dir, "stdin"), "utf8") };
}

/** Temporary synthetic home; null config means no file, raw permits invalid JSON fixtures. */
async function runnerWith(config, { raw, modelConfig } = {}) {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "agent-home-"));
  if (config !== null || raw !== undefined) {
    fs.mkdirSync(path.join(home, ".config", "eink"), { recursive: true });
    fs.writeFileSync(path.join(home, ".config", "eink", "agent.json"), raw ?? JSON.stringify(config));
  }
  if (modelConfig !== undefined) {
    fs.mkdirSync(path.join(home, ".codex"));
    fs.writeFileSync(path.join(home, ".codex", "config.toml"), modelConfig);
  }
  const inHome = (fn) => {
    const saved = process.env.HOME;
    process.env.HOME = home;
    try { return fn(); } finally {
      if (saved === undefined) delete process.env.HOME; else process.env.HOME = saved;
    }
  };
  const saved = process.env.HOME;
  process.env.HOME = home;
  let m;
  try { m = await import(`${path.join(__dirname, "..", "scripts", "agent-run.mjs")}?home=${encodeURIComponent(home)}`); }
  finally { if (saved === undefined) delete process.env.HOME; else process.env.HOME = saved; }
  return {
    agentConfig: () => inHome(() => m.agentConfig()),
    runAgent: (run) => inHome(() => m.runAgent(run)),
  };
}

const after = (a, flag, n = 1) => a.slice(a.indexOf(flag), a.indexOf(flag) + 1 + n);

test("agentConfig: configured Codex and explicit Codex environment override remain supported", async () => {
  const m = await runnerWith({ agent: "codex", model: "m1" });
  assert.equal(m.agentConfig().agent, "codex");
  process.env.EINK_AGENT = "codex";
  try { assert.equal(m.agentConfig().agent, "codex"); } finally { delete process.env.EINK_AGENT; }
});

const run = { prompt: "SYNTHETIC", cwd: os.tmpdir(), claudeTools: [] };
async function rejects(config, diagnostic, options) {
  const m = await runnerWith(config, options);
  const before = calls.length;
  const r = m.runAgent(run);
  assert.equal(calls.length, before, "invalid selection must spawn nothing, including Claude or Codex");
  assert.equal(r.status, 2);
  assert.equal(r.signal, null);
  assert.equal(r.stdout, "");
  assert.match(r.stderr, diagnostic);
  assert.match(r.stderr, /agent\.json/);
}

test("runAgent: missing, malformed, empty and non-object configurations fail closed", async (t) => {
  for (const [name, config, raw, diagnostic] of [
    ["missing", null, undefined, /configuration/i],
    ["malformed", null, "{", /JSON/i],
    ["empty file", null, "", /JSON/i],
    ["empty object", {}, undefined, /agent/i],
    ["null document", null, "null", /object/i],
    ["array", [], undefined, /object/i],
    ["string", "codex", undefined, /object/i],
    ["number", 1, undefined, /object/i],
    ["boolean", false, undefined, /object/i],
  ]) await t.test(name, () => rejects(config, diagnostic, { raw }));
});

test("runAgent: invalid provider types and unknown providers diagnose without spawning", async (t) => {
  const fake = fakeAgent();
  for (const agent of [null, "", " ", 1, false, [], {}, "nobody", "Codex"]) {
    await t.test(JSON.stringify(agent), () => rejects({ agent, command: fake.bin, model: "m1" }, /agent/i));
  }
});

test("runAgent: explicit Claude is disabled, even with a synthetic command or Codex override", async (t) => {
  const fake = fakeAgent();
  for (const override of [undefined, "codex"]) {
    await t.test(`override=${override}`, async () => {
      if (override !== undefined) process.env.EINK_AGENT = override;
      try { await rejects({ agent: "claude", command: fake.bin }, /Claude.*disabled|Codex.only/i); }
      finally { delete process.env.EINK_AGENT; }
    });
  }
});

test("runAgent: a Claude executable cannot masquerade as a Codex command", async (t) => {
  for (const command of ["claude", "/synthetic/bin/claude"]) {
    await t.test(command, () => rejects({ agent: "codex", model: "m1", command }, /Claude.*disabled|Codex.only/i));
  }
});

test("runAgent: Claude, empty and unknown environment overrides never fall back to configured Codex", async (t) => {
  const fake = fakeAgent();
  for (const agent of ["claude", "", " ", "nobody"]) {
    await t.test(JSON.stringify(agent), async () => {
      process.env.EINK_AGENT = agent;
      try { await rejects({ agent: "codex", model: "m1", command: fake.bin }, /Claude.*disabled|Codex.only|agent/i); }
      finally { delete process.env.EINK_AGENT; }
    });
  }
});

test("runAgent: invalid command and model values do not select native defaults", async (t) => {
  const fake = fakeAgent();
  for (const field of ["command", "model"]) {
    for (const value of [null, "", " ", 1, false, [], {}]) {
      await t.test(`${field}=${JSON.stringify(value)}`, () => rejects(
        { agent: "codex", model: "m1", command: fake.bin, [field]: value }, new RegExp(field, "i")));
    }
  }
});

test("runAgent: Codex gets the prompt on stdin, a sandbox, writable dirs, images, and its answer read back", async () => {
  const fake = fakeAgent();
  const { runAgent } = await runnerWith({ agent: "codex", model: "m1", command: fake.bin });
  const ro = runAgent({ prompt: "look", cwd: "/repo", images: ["/p.png"], claudeTools: ["Read"] });
  assert.equal(ro.status, 0);
  assert.equal(ro.agent, "codex");
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

test("runAgent: no resolved model fails closed instead of choosing a native model", async () => {
  await rejects({ agent: "codex", command: fakeAgent().bin }, /model/i);
});

test("runAgent: synthetic model plumbing preserves explicit model and configured TOML model", async () => {
  const fake = fakeAgent();
  for (const model of [undefined, "exact-config-model"]) {
    const { runAgent } = await runnerWith({ agent: "codex", command: fake.bin, ...(model ? { model } : {}) },
      { modelConfig: 'model = "exact-toml-model"\n' });
    assert.equal(runAgent(run).status, 0);
    assert.deepEqual(after(fake.args(), "-m"), ["-m", model ?? "exact-toml-model"]);
  }
});

test("runAgent: executable selection remains exact (intercepted, never native)", async () => {
  const bundled = "/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex";
  const { runAgent } = await runnerWith({ agent: "codex", model: "m1" });
  const before = calls.length;
  assert.equal(runAgent(run).status, 97, "synthetic interceptor blocks native executable");
  assert.equal(calls.length, before + 1);
  assert.equal(calls.at(-1).command, fs.existsSync(bundled) ? bundled : "codex");
});

test("runAgent: synthetic nonzero and spawn failures propagate without any retry or fallback", async () => {
  for (const missing of [false, true]) {
    const fake = fakeAgent(42);
    if (missing) fs.rmSync(fake.bin);
    const { runAgent } = await runnerWith({ agent: "codex", model: "exact-model", command: fake.bin });
    const before = calls.length;
    const r = runAgent(run);
    assert.equal(calls.length, before + 1);
    assert.equal(calls.at(-1).command, fake.bin);
    assert.deepEqual(after(calls.at(-1).args, "-m"), ["-m", "exact-model"]);
    assert.equal(r.agent, "codex");
    if (missing) {
      assert.equal(r.status, null);
      assert.equal(r.error.code, "ENOENT");
      assert.equal(r.stdout, "");
    } else {
      assert.equal(r.status, 42);
      assert.equal(r.stderr, "SYNTHETIC stderr");
    }
  }
});

test("runAgent: zero Claude command attempts across synthetic controls", () => {
  assert.equal(calls.filter(({ command }) => path.basename(String(command)) === "claude").length, 0);
});
