/**
 * Adapter for Jarvis's staged family-photo pool (DECISIONS #106).
 *
 * The Mac writes an immutable set under `family/staged/{date}/` and puts `photo.json` last,
 * only after it has read the listed bytes back. This module is the only place the Worker
 * interprets that manifest: it lists, checks hashes, claims with the object's R2 ETag, and
 * marks an entry used. It never deletes staged objects. The Mac owns their lifecycle.
 */

import type { Env } from "./types";

const STAGED_PREFIX = "family/staged/";
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const SHA_RE = /^[0-9a-f]{64}$/i;

export const stagedManifestKey = (date: string) => `family/staged/${date}/photo.json`;
export const stagedPhotoKey = (date: string) => `family/staged/${date}/photo.jpg`;
export const stagedRepaintKey = (date: string) => `family/staged/${date}/repaint.png`;

export interface StagedManifest {
  schema: 1;
  /** Intended Chicago display date. The object key uses the same date. */
  date: string;
  taken: string;
  place: string;
  stagedAt: string;
  photoSha256: string;
  repaintSha256: string | null;
  used: boolean;
  usedFor: string | null;
  usedAt: string | null;
  claimFor: string | null;
}

export interface StagedRecord {
  key: string;
  /** Parsed JSON object, kept so a claim write does not drop fields the Mac added. */
  raw: Record<string, unknown>;
  httpEtag: string;
  manifest: StagedManifest;
}

export interface StagedAsset {
  kind: "repaint" | "photo";
  bytes: Uint8Array;
}

export interface StagedCacheMarker {
  entry: string;
  asset: "repaint" | "photo";
}

export function stagedDateFromKey(key: string): string | null {
  const match = /^family\/staged\/(\d{4}-\d{2}-\d{2})\/photo\.json$/.exec(key);
  return match ? match[1] : null;
}

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function asDate(value: unknown): string | null {
  return typeof value === "string" && DATE_RE.test(value) ? value : null;
}

function asSha(value: unknown): string | null {
  return typeof value === "string" && SHA_RE.test(value) ? value.toLowerCase() : null;
}

/** A manifest is real only when it matches its key and the fields the Mac promised. */
export function parseStagedManifest(raw: unknown, keyDate: string): StagedManifest | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const o = raw as Record<string, unknown>;
  if (o.schema !== 1) return null;
  if (o.date !== keyDate || !asDate(keyDate)) return null;
  const taken = asDate(o.taken);
  if (!taken) return null;
  if (typeof o.place !== "string") return null;
  if (typeof o.stagedAt !== "string" || !Number.isFinite(Date.parse(o.stagedAt))) return null;
  const photoSha256 = asSha(o.photoSha256);
  if (!photoSha256) return null;
  let repaintSha256: string | null;
  if (o.repaintSha256 === null) repaintSha256 = null;
  else {
    const sha = asSha(o.repaintSha256);
    if (!sha) return null;
    repaintSha256 = sha;
  }
  if (typeof o.used !== "boolean") return null;
  const usedFor = o.usedFor === null || o.usedFor === undefined ? null : asDate(o.usedFor);
  if (o.usedFor !== null && o.usedFor !== undefined && !usedFor) return null;
  if (o.usedAt !== null && o.usedAt !== undefined && typeof o.usedAt !== "string") return null;
  const claimFor = o.claimFor === null || o.claimFor === undefined ? null : asDate(o.claimFor);
  if (o.claimFor !== null && o.claimFor !== undefined && !claimFor) return null;
  return {
    schema: 1,
    date: keyDate,
    taken,
    place: o.place,
    stagedAt: o.stagedAt,
    photoSha256,
    repaintSha256,
    used: o.used,
    usedFor,
    usedAt: typeof o.usedAt === "string" ? o.usedAt : null,
    claimFor,
  };
}

export function readStagedMarker(raw: string | null): StagedCacheMarker | null {
  if (!raw || !raw.startsWith("{")) return null;
  try {
    const staged = (JSON.parse(raw) as { staged?: { entry?: unknown; asset?: unknown } }).staged;
    if (!staged || !asDate(staged.entry)) return null;
    if (staged.asset !== "repaint" && staged.asset !== "photo") return null;
    return { entry: String(staged.entry), asset: staged.asset };
  } catch {
    return null;
  }
}

/** `drawn_by.family` when the cached picture came from the staged pool. */
export function familyCacheSource(raw: string | null): "staged" | "staged-photo" | null {
  const marker = readStagedMarker(raw);
  if (!marker) return null;
  return marker.asset === "repaint" ? "staged" : "staged-photo";
}

function manifestBody(raw: Record<string, unknown>, patch: Record<string, unknown>): string {
  return JSON.stringify({ ...raw, ...patch });
}

export async function readStagedManifest(env: Env, key: string): Promise<StagedRecord | null> {
  const date = stagedDateFromKey(key);
  if (!date) return null;
  const obj = await env.PHOTOS.get(key);
  if (!obj) return null;
  const httpEtag = obj.httpEtag || obj.etag;
  if (!httpEtag) return null;
  let raw: unknown;
  try { raw = await obj.json(); } catch { return null; }
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const manifest = parseStagedManifest(raw, date);
  if (!manifest) return null;
  return { key, raw: raw as Record<string, unknown>, httpEtag, manifest };
}

/** Every manifest currently in the pool. A missing `list` (unit-test buckets) is an empty pool. */
export async function listStagedManifests(env: Env): Promise<StagedRecord[]> {
  if (typeof env.PHOTOS.list !== "function") return [];
  const keys: string[] = [];
  let cursor: string | undefined;
  for (let page = 0; page < 20; page++) {
    const listed = await env.PHOTOS.list(cursor ? { prefix: STAGED_PREFIX, cursor } : { prefix: STAGED_PREFIX });
    for (const obj of listed.objects) {
      if (obj.key.endsWith("/photo.json")) keys.push(obj.key);
    }
    if (!listed.truncated) break;
    cursor = listed.cursor;
  }
  const records: StagedRecord[] = [];
  for (const key of keys) {
    const record = await readStagedManifest(env, key);
    if (record) records.push(record);
  }
  return records;
}

function stagedAtMs(value: string): number {
  const n = Date.parse(value);
  return Number.isFinite(n) ? n : 0;
}

/**
 * Fresh picks only: unused and unclaimed. The display date's own entry comes first.
 * Anything else is newest `stagedAt`, then the later intended date when those tie.
 * An outstanding claim is not in this list; the caller reconciles its own claim apart.
 */
export function stagedPickOrder(records: readonly StagedRecord[], displayDate: string): StagedRecord[] {
  const available = records.filter((r) => !r.manifest.used && r.manifest.claimFor === null);
  const exact = available.filter((r) => r.manifest.date === displayDate);
  const rest = available.filter((r) => r.manifest.date !== displayDate);
  rest.sort((a, b) => {
    const byTime = stagedAtMs(b.manifest.stagedAt) - stagedAtMs(a.manifest.stagedAt);
    if (byTime !== 0) return byTime;
    return b.manifest.date.localeCompare(a.manifest.date);
  });
  return [...exact, ...rest];
}

export function claimsFor(records: readonly StagedRecord[], displayDate: string): StagedRecord[] {
  return records.filter((r) => !r.manifest.used && r.manifest.claimFor === displayDate);
}

/**
 * Conditional claim. A lost ETag race is re-read, so a repaint the Mac added before the
 * claim is the manifest we keep. Returns null when the entry is used or held by another date.
 */
export async function claimStaged(env: Env, key: string, displayDate: string): Promise<StagedRecord | null> {
  for (let attempt = 0; attempt < 3; attempt++) {
    const current = await readStagedManifest(env, key);
    if (!current || current.manifest.used) return null;
    if (current.manifest.claimFor && current.manifest.claimFor !== displayDate) return null;
    if (current.manifest.claimFor === displayDate) return current;
    const written = await env.PHOTOS.put(key, manifestBody(current.raw, { claimFor: displayDate }), {
      onlyIf: { etagMatches: current.httpEtag },
      httpMetadata: { contentType: "application/json" },
    });
    if (!written) continue;
    const again = await readStagedManifest(env, key);
    if (again?.manifest.claimFor === displayDate) return again;
  }
  return null;
}

export async function releaseStagedClaim(env: Env, key: string, displayDate: string): Promise<void> {
  for (let attempt = 0; attempt < 3; attempt++) {
    const current = await readStagedManifest(env, key);
    if (!current || current.manifest.used || current.manifest.claimFor !== displayDate) return;
    const written = await env.PHOTOS.put(key, manifestBody(current.raw, { claimFor: null }), {
      onlyIf: { etagMatches: current.httpEtag },
      httpMetadata: { contentType: "application/json" },
    });
    if (written) return;
  }
}

/** Drop this display date's claims that were never marked used. A dated fill calls this. */
export async function releaseStagedClaimsFor(env: Env, displayDate: string): Promise<void> {
  const records = await listStagedManifests(env);
  for (const record of claimsFor(records, displayDate)) {
    await releaseStagedClaim(env, record.key, displayDate);
  }
}

export async function markStagedUsed(env: Env, key: string, displayDate: string, usedAt: string): Promise<boolean> {
  for (let attempt = 0; attempt < 3; attempt++) {
    const current = await readStagedManifest(env, key);
    if (!current) return false;
    if (current.manifest.used && current.manifest.usedFor === displayDate) return true;
    if (current.manifest.claimFor !== displayDate || current.manifest.used) return false;
    const written = await env.PHOTOS.put(key, manifestBody(current.raw, {
      used: true,
      usedFor: displayDate,
      usedAt,
      claimFor: displayDate,
    }), {
      onlyIf: { etagMatches: current.httpEtag },
      httpMetadata: { contentType: "application/json" },
    });
    if (written) return true;
  }
  return false;
}

/**
 * Bytes the manifest names, checked before use. A photo hash miss, or a declared repaint
 * that is missing or mismatched, skips the whole entry. `repaintSha256: null` means the
 * photo itself is the asset.
 */
export async function loadVerifiedStagedAsset(env: Env, manifest: StagedManifest): Promise<StagedAsset | null> {
  const photoObj = await env.PHOTOS.get(stagedPhotoKey(manifest.date));
  if (!photoObj) return null;
  const photoBytes = new Uint8Array(await photoObj.arrayBuffer());
  if ((await sha256Hex(photoBytes)) !== manifest.photoSha256) return null;
  if (manifest.repaintSha256) {
    const repaintObj = await env.PHOTOS.get(stagedRepaintKey(manifest.date));
    if (!repaintObj) return null;
    const repaintBytes = new Uint8Array(await repaintObj.arrayBuffer());
    if ((await sha256Hex(repaintBytes)) !== manifest.repaintSha256) return null;
    return { kind: "repaint", bytes: repaintBytes };
  }
  return { kind: "photo", bytes: photoBytes };
}

/** Unused, unclaimed manifests. Health uses this count; it does not re-hash the bytes. */
export async function countUnusedStaged(env: Env): Promise<number | null> {
  try {
    const records = await listStagedManifests(env);
    return records.filter((r) => !r.manifest.used && r.manifest.claimFor === null).length;
  } catch (err) {
    console.error("Family staged pool count failed:", err);
    return null;
  }
}
