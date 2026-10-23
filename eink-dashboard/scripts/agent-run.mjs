/**
 * The one place the Mac mini's jobs start an unattended AI run (DECISIONS #98): the serial's
 * writer and reviewer, and the series mailbox's message, chapter and review runs. Which agent runs
 * them comes from ~/.config/eink/agent.json, or EINK_AGENT for a single run:
 *
 *   {"agent": "claude"}                                        the default when nothing is set
 *   {"agent": "codex", "model": "gpt-6.1-sol", "command": "/path/to/codex"}
 *
 * Every run gets a fixed instruction and narrow rights. Claude enforces `claudeTools` per tool.
 * Codex has no per-tool rules, so it gets the nearest sandbox: read-only when nothing is written,
 * else writable only in `write` (its working root is the first of them, never the repo). It runs
 * without the user's Codex config, so its memories never keep a story it read.
 * Its model is agent.json's `model`, else the `model =` line of ~/.codex/config.toml (the one the
 * ChatGPT app uses): Codex's built-in default is refused on a ChatGPT sign-in.
 * Returns {status, stdout, stderr, signal, agent}: stdout is the agent's final answer.
 */
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const CONFIG = path.join(os.homedir(), ".config", "eink", "agent.json");
// The ChatGPT app's own CLI: the Homebrew one lags it and is refused the account's models.
const BUNDLED_CODEX = "/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex";

function codexModel() {
  try { return /^model\s*=\s*"([^"]+)"/m.exec(fs.readFileSync(path.join(os.homedir(), ".codex", "config.toml"), "utf8"))?.[1]; } catch { return undefined; }
}

export function agentConfig() {
  let file = {};
  try { file = JSON.parse(fs.readFileSync(CONFIG, "utf8")); } catch { /* no config: Claude */ }
  return { ...file, agent: process.env.EINK_AGENT ?? file.agent ?? "claude" };
}

/**
 * @param {object} run
 * @param {string} run.prompt       the instruction, placeholders already filled
 * @param {string} run.cwd          where Claude runs (Codex runs in write[0] when it writes)
 * @param {string[]} [run.read]     directories it reads besides cwd (Codex reads anywhere)
 * @param {string[]} [run.write]    directories it may write
 * @param {string[]} [run.images]   pictures it must look at (attached for Codex)
 * @param {string[]} run.claudeTools the exact Claude tool rules
 * @param {boolean} [run.web]       it may search the web
 */
export function runAgent({ prompt, cwd, read = [], write = [], images = [], claudeTools, web = false, timeout, maxBuffer = 64 * 1024 * 1024 }) {
  const cfg = agentConfig();
  if (cfg.agent === "claude") {
    const dirs = [...new Set([...read, ...write])].flatMap((d) => ["--add-dir", d]);
    const r = spawnSync(cfg.command ?? "claude", ["-p", prompt, "--model", cfg.model ?? "opus", "--permission-mode", "dontAsk",
      "--allowedTools", claudeTools.join(" "), ...dirs], { cwd, encoding: "utf8", env: process.env, timeout, maxBuffer });
    return { ...r, agent: "claude" };
  }
  if (cfg.agent === "codex") {
    const outDir = fs.mkdtempSync(path.join(os.tmpdir(), "agent-out-"));
    const answer = path.join(outDir, "answer.txt");
    const model = cfg.model ?? codexModel();
    const args = ["exec", "--ignore-user-config", "--skip-git-repo-check", "--ephemeral", "--color", "never",
      ...(model ? ["-m", model] : []),
      "-s", write.length ? "workspace-write" : "read-only", "-C", write[0] ?? cwd,
      ...write.slice(1).flatMap((d) => ["--add-dir", d]),
      ...(web ? ["-c", "sandbox_workspace_write.network_access=true", "-c", 'web_search="live"'] : []),
      ...images.flatMap((f) => ["-i", f]),
      "-o", answer, "-"];
    const command = cfg.command ?? (fs.existsSync(BUNDLED_CODEX) ? BUNDLED_CODEX : "codex");
    const r = spawnSync(command, args, { input: prompt, encoding: "utf8", env: process.env, timeout, maxBuffer });
    const stdout = fs.existsSync(answer) ? fs.readFileSync(answer, "utf8") : "";
    fs.rmSync(outDir, { recursive: true, force: true });
    return { ...r, stdout, agent: "codex" };
  }
  return { status: 2, signal: null, stdout: "", stderr: `unknown agent "${cfg.agent}" in ${CONFIG}`, agent: cfg.agent };
}
