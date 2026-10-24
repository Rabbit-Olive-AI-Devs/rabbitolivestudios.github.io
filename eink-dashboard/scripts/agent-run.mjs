/**
 * The one place the Mac mini's jobs start an unattended AI run (DECISIONS #98): the serial's
 * writer and reviewer, and the series mailbox's message, chapter and review runs. Which agent runs
 * them comes from ~/.config/eink/agent.json, or EINK_AGENT for a single run:
 *
 *   {"agent": "codex", "model": "gpt-6.1-sol", "command": "/path/to/codex"}
 *
 * Only Codex is allowed. Missing/invalid configuration and explicit Claude selection fail closed.
 * Every run gets a fixed instruction and the native sandbox: read-only when nothing is written,
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
  let file;
  const invalid = (message, agent) => ({ agent, error: `${message} in ${CONFIG}` });
  try { file = JSON.parse(fs.readFileSync(CONFIG, "utf8")); }
  catch (error) { return invalid(error instanceof SyntaxError ? "invalid configuration JSON" : "unreadable agent configuration"); }
  if (!file || typeof file !== "object" || Array.isArray(file)) return invalid("agent configuration must be an object");
  // An override must not conceal a stale/invalid provider configuration.
  for (const agent of [file.agent, process.env.EINK_AGENT]) {
    if (agent === "claude") return invalid("Claude is disabled; this project is Codex-only", agent);
    if (agent !== undefined && agent !== "codex") return invalid(`unknown agent ${JSON.stringify(agent)}; expected "codex"`, agent);
  }
  const agent = process.env.EINK_AGENT ?? file.agent;
  if (agent !== "codex") return invalid('agent must be explicitly "codex"', agent);
  for (const key of ["command", "model"]) {
    if (Object.hasOwn(file, key) && (typeof file[key] !== "string" || !file[key].trim())) {
      return invalid(`${key} must be a non-empty string`, agent);
    }
  }
  if (file.command && path.basename(file.command) === "claude") return invalid("Claude command is disabled; this project is Codex-only", agent);
  return { ...file, agent, error: undefined };
}

/**
 * @param {object} run
 * @param {string} run.prompt       the instruction, placeholders already filled
 * @param {string} run.cwd          Codex's working root when there are no writes
 * @param {string[]} [run.read]     legacy hint only (Codex reads anywhere)
 * @param {string[]} [run.write]    directories it may write
 * @param {string[]} [run.images]   pictures it must look at (attached for Codex)
 * @param {string[]} [run.claudeTools] legacy caller compatibility; unused
 * @param {boolean} [run.web]       it may search the web
 */
export function runAgent({ prompt, cwd, read = [], write = [], images = [], claudeTools, web = false, timeout, maxBuffer = 64 * 1024 * 1024 }) {
  const cfg = agentConfig();
  if (cfg.error) return { status: 2, signal: null, stdout: "", stderr: cfg.error, agent: cfg.agent };
  if (cfg.agent === "codex") {
    const model = cfg.model ?? codexModel();
    if (!model?.trim()) return { status: 2, signal: null, stdout: "", stderr: `no Codex model configured in ${CONFIG} or ~/.codex/config.toml`, agent: cfg.agent };
    const outDir = fs.mkdtempSync(path.join(os.tmpdir(), "agent-out-"));
    const answer = path.join(outDir, "answer.txt");
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
