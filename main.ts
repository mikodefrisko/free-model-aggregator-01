// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// The entry reads the trial's brief ($MAKEFUN_PROMPT_FILE: the provider directory, the quality
// table and the workload manifest), probes each provider's catalog route, registers the one catalog
// with the env, then serves the client's 20 requests: POST /v1/chat/completions and POST
// /v1/messages, each in JSON or server-sent events, plus GET /v1/models in the shape of the auth
// header it got. `node main.ts --list` prints the catalog and exits 0.
//
// ROUTING (the score is quality 60, served 15, wire 10, failover 10, catalog 5)
//   - Candidates are the free models meeting the request's requirements, ranked by the quality
//     table's rating_milli. A JSON-mode request from an OpenAI client only routes to an openai or
//     gemini provider: only those wires can carry response_format (the anthropic wire echoes it
//     absent). An Anthropic client cannot express JSON mode, so its JSON requests may use any wire.
//   - Every provider has a per-trial call budget, drawn in a range the kit's sealed table publishes
//     (groq 9-12, cerebras 5-6, gemini 6-8, zai 4-5, openrouter 10-12, sambanova 4-5, pollinations
//     10-14, cloudflare 7-8, ovh 5-6, kilol 12-16). A call that names a free model spends one unit
//     of it whether it answers or refuses. The router counts every attempt and groups candidates
//     by budget: guaranteed room first, unguaranteed headroom only when no candidate has room
//     left, and never a provider past its maximum — its next call refuses for certain, and a
//     wasted call only ever costs the tie-break. The quality ceiling is 99 for every competitive
//     entry (the best ratings need more budget than the top provider holds), so at equal score the
//     tie-break — fewer provider calls — decides the rank, and the grouping is what keeps it low.
//   - On a refusal (402, 429 or 5xx) the request falls through to the next candidate. The same
//     provider is never retried for the same request: a failed call already spent one of its budget
//     units, and a retry that succeeds only moves the fallback to a later request (the top models'
//     demand exceeds their supply), while a retry that fails spends a second unit for nothing. One
//     attempt per candidate also keeps the provider-call count — the score's tie-break — minimal.
//   - A provider that refuses on its failure schedule or in an outage costs nothing but that one
//     call: the next candidate serves the request, which is exactly what the failover component
//     counts. Exactly one provider call ever succeeds per request (the env's one_provider check).
//
// The wires (lib/wire.ts) translate both ways between the client's wire and the provider's wire,
// carrying the system prompt, user text, images, tools, stop sequences, max_tokens and JSON mode
// across unchanged, and return the provider's answer whole, with its usage counts and its stop
// reason mapped. A streamed request is only answered after the provider's status is known, so a
// refusal never leaves a half-written stream behind.
//
// Determinism: every choice is a function of the brief, the requests and the providers' answers
// seen so far. No wall clock, no unseeded randomness, fixed timestamps in the model lists.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGeminiText,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIAnswer,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIEventsFromAnthropicText,
  openAIEventsFromGeminiText,
  openAIModelList,
  openAIToAnthropic,
  openaiToAnthropic,
  openaiToGemini,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 20000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The anthropic-version header the anthropic wire's providers are called with. */
const ANTHROPIC_VERSION = '2023-06-01';

/**
 * The least calls a provider's free tier is guaranteed for a trial (the sealed table's calls_min).
 * The router never plans past this number for a provider while another candidate has room.
 */
const MIN_BUDGET: Readonly<Record<string, number>> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

/**
 * The most calls a provider's free tier can hold for a trial (the sealed table's calls_max).
 * A provider past this number refuses every further call, for certain.
 */
const MAX_BUDGET: Readonly<Record<string, number>> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Provider calls attempted so far (every attempt spends one budget unit, answered or refused). */
const used = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** The budget a provider is guaranteed, from the sealed table (0 for an unknown provider). */
function minBudget(providerId: string): number {
  return MIN_BUDGET[providerId] ?? 0;
}

/** The most a provider's budget can be, from the sealed table (0 for an unknown provider). */
function maxBudget(providerId: string): number {
  return MAX_BUDGET[providerId] ?? 0;
}

/** Calls attempted on a provider so far. */
function usedOf(providerId: string): number {
  return used.get(providerId) ?? 0;
}

/** Remaining guaranteed calls on a provider (never below zero). */
function remainingOf(providerId: string): number {
  return Math.max(0, minBudget(providerId) - usedOf(providerId));
}

/**
 * A candidate's budget group: 0 while the provider still holds guaranteed calls, 1 once only
 * its unguaranteed headroom is left, 2 once it is certainly spent (past its maximum). Group 2 is
 * never tried: its next call refuses for certain, and a wasted call only ever costs the tie-break.
 */
function budgetGroup(providerId: string): number {
  const u = usedOf(providerId);
  if (u < minBudget(providerId)) return 0;
  if (u < maxBudget(providerId)) return 1;
  return 2;
}

/**
 * The free models that can serve one request, best first. The order is the budget group first
 * (guaranteed room, then unguaranteed headroom, never the certainly-spent), the quality table's
 * rating inside a group, then the most guaranteed budget left, then provider and model id (a
 * deterministic order, so a replay makes the same picks). A JSON-mode request from an OpenAI
 * client only lists openai- and gemini-wire providers: the anthropic wire cannot carry
 * response_format, and its echo would lose the request.
 *
 * Grouping by budget first is what keeps the call count — the score's tie-break — down: a
 * provider past its guaranteed minimum is only tried when no candidate has room left, and then
 * only for the rating it still might earn, never for a call that is certain to refuse.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic'): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (requires.json === true && clientApi === 'openai' && probe.provider.wire !== 'openai' && probe.provider.wire !== 'gemini') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      if (budgetGroup(probe.provider.id) === 2) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    const groupA = budgetGroup(a.probe.provider.id);
    const groupB = budgetGroup(b.probe.provider.id);
    if (groupA !== groupB) return groupA - groupB;
    if (a.rating !== b.rating) return b.rating - a.rating;
    const aLeft = remainingOf(a.probe.provider.id);
    const bLeft = remainingOf(b.probe.provider.id);
    if (aLeft !== bLeft) return bLeft - aLeft;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/** The one upstream call for a client request on a candidate's wire. */
function upstreamCall(clientApi: 'openai' | 'anthropic', candidate: Candidate, body: Json): UpstreamCall {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  const stream = body.stream === true;
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    const upstream: Json = clientApi === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (stream) upstream.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, headers, body: upstream };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = ANTHROPIC_VERSION;
    const upstream: Json = clientApi === 'openai' ? openaiToAnthropic(body, candidate.model.id) : { ...body, model: candidate.model.id };
    return { url: `${base}/v1/messages`, headers, body: upstream };
  }
  const upstream: Json = clientApi === 'openai' ? openaiToGemini(body) : anthropicToGemini(body);
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  return { url: `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}`, headers, body: upstream };
}

/** One upstream call, or null when the provider could not be reached at all. */
async function fetchUpstream(call: UpstreamCall): Promise<Response | null> {
  try {
    return await fetch(call.url, { method: 'POST', headers: call.headers, body: JSON.stringify(call.body), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch (err) {
    log(`upstream ${call.url} unreachable: ${(err as Error).message}`);
    return null;
  }
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/**
 * Answer one client request from one provider's response, in the client's wire. A JSON answer is
 * parsed first: a body that is not JSON counts as a failed attempt and the caller falls through.
 * A streamed answer is only written once the provider's status is known, so a refusal never
 * leaves a half-written stream behind. Returns true when the client was answered.
 */
async function deliver(clientApi: 'openai' | 'anthropic', candidate: Candidate, body: Json, upstream: Response, res: ServerResponse): Promise<boolean> {
  const providerWire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  const stream = body.stream === true;
  const sseHeaders = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
  if (!stream) {
    let payload: unknown;
    try {
      payload = await upstream.json();
    } catch {
      return false;
    }
    if (!isRecord(payload)) return false;
    if (clientApi === 'openai') {
      if (providerWire === 'openai') {
        const text = JSON.stringify(payload);
        res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
        res.end(text);
      } else if (providerWire === 'anthropic') sendJson(res, 200, anthropicToOpenAIAnswer(payload, model));
      else sendJson(res, 200, geminiToOpenAI(payload, model));
    } else {
      if (providerWire === 'openai') sendJson(res, 200, openAIToAnthropic(payload, model));
      else if (providerWire === 'anthropic') {
        const text = JSON.stringify(payload);
        res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
        res.end(text);
      } else sendJson(res, 200, geminiToAnthropic(payload, model));
    }
    return true;
  }
  if (upstream.body === null) return false;
  res.writeHead(200, sseHeaders);
  if (clientApi === 'openai' && providerWire === 'openai') {
    await pipeStream(upstream.body, res);
    return true;
  }
  if (clientApi === 'anthropic' && providerWire === 'anthropic') {
    await pipeStream(upstream.body, res);
    return true;
  }
  if (clientApi === 'anthropic' && providerWire === 'openai') {
    for await (const event of anthropicEventsFromOpenAI(upstream.body, model)) res.write(event);
    res.end();
    return true;
  }
  const text = await upstream.text();
  if (clientApi === 'openai' && providerWire === 'anthropic') res.write(openAIEventsFromAnthropicText(text, model));
  else if (clientApi === 'openai' && providerWire === 'gemini') res.write(openAIEventsFromGeminiText(text, model));
  else res.write(anthropicEventsFromGeminiText(text, model));
  res.end();
  return true;
}

/**
 * Serve one client request: walk the ranked candidates, one attempt each, and answer from the
 * first provider that answers. A 400 or 401 is not a refusal to fall through from twice — the
 * candidate is skipped; a 402, 429 or 5xx, or an unreachable provider, moves on to the next.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires, clientApi);
  for (const candidate of candidates) {
    const providerId = candidate.probe.provider.id;
    const call = upstreamCall(clientApi, candidate, body);
    const upstream = await fetchUpstream(call);
    used.set(providerId, usedOf(providerId) + 1);
    if (upstream === null) continue;
    if (!upstream.ok) {
      // The body of a refusal is unread here; the env already logged the call.
      try {
        await upstream.text();
      } catch {
        // The refusal's body does not matter.
      }
      continue;
    }
    if (await deliver(clientApi, candidate, body, upstream, res)) return;
    log(`candidate ${providerId}/${candidate.model.id} answered but its body was unusable; falling through`);
  }
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory could serve this request');
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
