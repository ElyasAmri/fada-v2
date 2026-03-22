/**
 * FADA RCCG Channel Server
 *
 * One-way Claude Code channel plugin that receives webhook POSTs from
 * fada-monitor on RCCG machines and pushes them into the Claude Code
 * session as channel events.
 *
 * Architecture:
 *   fada-monitor (RCCG) -> HTTP POST -> Cloudflare Tunnel
 *     -> localhost:8789 (this server) -> MCP notification -> Claude Code
 */
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { readFileSync, existsSync } from "fs";
import { join } from "path";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const PORT = parseInt(process.env.FADA_CHANNEL_PORT || "8789", 10);

function loadToken(): string {
  // Env var takes precedence
  if (process.env.FADA_CHANNEL_TOKEN) {
    return process.env.FADA_CHANNEL_TOKEN;
  }
  // Fall back to the token file created by deploy_monitor playbook
  const tokenPath = join(
    process.env.USERPROFILE || process.env.HOME || "",
    ".claude",
    "hooks",
    ".monitor_token"
  );
  if (existsSync(tokenPath)) {
    return readFileSync(tokenPath, "utf-8").trim();
  }
  console.error(
    "[fada-channel] WARNING: No auth token configured. " +
      "Set FADA_CHANNEL_TOKEN or deploy monitor to create .monitor_token"
  );
  return "";
}

const AUTH_TOKEN = loadToken();

// ---------------------------------------------------------------------------
// Rate limiting: priority-based per hostname
// Critical events (process_completed, process_started) are never rate-limited.
// Idle events are limited to 1 per hostname per 5 minutes.
// ---------------------------------------------------------------------------
const idleRateLimits = new Map<string, number>();
const IDLE_RATE_LIMIT_MS = 300_000; // 5 minutes

const CRITICAL_EVENTS = new Set(["process_completed", "process_started"]);

function checkRateLimit(hostname: string, kind: string): boolean {
  // Critical events are never rate-limited
  if (CRITICAL_EVENTS.has(kind)) {
    return true;
  }
  // Idle events: 1 per hostname per 5 minutes
  if (kind === "idle") {
    const now = Date.now();
    const last = idleRateLimits.get(hostname) || 0;
    if (now - last < IDLE_RATE_LIMIT_MS) {
      return false;
    }
    idleRateLimits.set(hostname, now);
    return true;
  }
  // Other events: allow (rare enough to not need limiting)
  return true;
}

// ---------------------------------------------------------------------------
// MCP Server (one-way channel, no reply tools)
// ---------------------------------------------------------------------------
const INSTRUCTIONS = `Events from fada-monitor on RCCG training machines arrive as <channel source="fada-rccg" ...> tags.

Event kinds:
- process_completed: A training/eval job finished. Attributes include hostname, pattern (script name), runtime_human.
- process_started: A new job was detected. Attributes include hostname, pattern, pid.
- idle: Machine has no watched processes running. Sent every 2 minutes while idle.

When you receive a process_completed event:
1. Pull results from the machine: ./r.sh pull <hostname>
2. Score the results if applicable
3. Check if there are queued models to launch next: ./r.sh status
4. Launch the next job if the machine is idle

When you receive an idle event:
1. The machine has no training/eval jobs running -- launch the next untrained model on it.

These are one-way notifications. No reply mechanism exists.`;

const mcp = new Server(
  { name: "fada-rccg", version: "1.0.0" },
  {
    capabilities: {
      experimental: { "claude/channel": {} },
    },
    instructions: INSTRUCTIONS,
  }
);

// ---------------------------------------------------------------------------
// HTTP Webhook Server
// ---------------------------------------------------------------------------
function formatEventContent(body: WebhookBody): string {
  const data = body.data || {};
  const hostname = data.hostname || "unknown";
  const pattern = data.pattern || "";
  const runtime = data.runtime_human || "";

  switch (body.kind) {
    case "process_completed":
      return `Job completed on ${hostname}: ${pattern} (runtime: ${runtime})\n\nACTION REQUIRED:\n1. Pull results: ./r.sh ssh ${hostname} "cat /home/ubuntu/fada-results/*/eval/checkpoint_*.json" > checkpoint.json\n2. Score: ./venv/Scripts/python.exe experiments/evaluation/checkpoint_to_predictions.py then score_against_gt.py\n3. Clean disk: ./r.sh ssh ${hostname} "rm -rf ~/.cache/huggingface/hub/models--*"\n4. Launch next untrained model immediately using experiments/rccg/launch_job.sh\n5. NEVER leave the GPU idle.`;
    case "process_started":
      return `Job started on ${hostname}: ${pattern} (pid: ${data.pid || "?"})`;
    case "idle":
      return `Machine ${hostname} is IDLE — GPU is being WASTED.\n\nACTION REQUIRED: Launch a new model IMMEDIATELY using experiments/rccg/launch_job.sh. Check llamafactory template compatibility. Clean disk first if needed. If all known-compatible models are exhausted, check llamafactory's template list for untried templates, or try models that previously failed on other machines with different errors. NEVER leave a GPU idle.`;
    default:
      return `Event [${body.kind}] on ${hostname}: ${JSON.stringify(data)}`;
  }
}

interface WebhookBody {
  id?: string;
  kind: string;
  data?: Record<string, unknown>;
  ts?: number;
  ts_iso?: string;
}

const httpServer = Bun.serve({
  port: PORT,
  hostname: "127.0.0.1",

  async fetch(req: Request): Promise<Response> {
    const url = new URL(req.url);

    // Health check (no auth required)
    if (req.method === "GET" && url.pathname === "/health") {
      return Response.json({ status: "ok", port: PORT });
    }

    // Webhook endpoint
    if (req.method === "POST" && url.pathname === "/webhook") {
      // Auth check
      if (AUTH_TOKEN) {
        const auth = req.headers.get("authorization") || "";
        if (!auth.startsWith("Bearer ")) {
          return Response.json({ error: "missing bearer token" }, { status: 401 });
        }
        const token = auth.slice(7);
        if (token !== AUTH_TOKEN) {
          return Response.json({ error: "invalid token" }, { status: 403 });
        }
      }

      // Parse body
      let body: WebhookBody;
      try {
        body = (await req.json()) as WebhookBody;
      } catch {
        return Response.json({ error: "invalid JSON" }, { status: 400 });
      }

      if (!body.kind) {
        return Response.json({ error: "missing 'kind' field" }, { status: 400 });
      }

      // Rate limit by hostname and event kind
      const hostname = (body.data?.hostname as string) || "unknown";
      if (!checkRateLimit(hostname, body.kind)) {
        return Response.json(
          { error: "rate limited", hostname, kind: body.kind },
          { status: 429 }
        );
      }

      // Build meta attributes (only simple alphanumeric/underscore keys)
      const meta: Record<string, string> = {};
      if (body.data) {
        for (const [key, val] of Object.entries(body.data)) {
          if (/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(key) && val != null) {
            meta[key] = String(val);
          }
        }
      }
      meta.kind = body.kind;
      if (body.id) meta.event_id = body.id;

      // Push to Claude Code session (best-effort -- polling fallback catches misses)
      const content = formatEventContent(body);
      let delivered = false;
      try {
        await mcp.notification({
          method: "notifications/claude/channel",
          params: { content, meta },
        });
        delivered = true;
      } catch (err) {
        console.error("[fada-channel] MCP notification failed (Claude likely not connected):", err);
      }

      console.error(
        `[fada-channel] ${delivered ? "Forwarded" : "Accepted (not delivered)"} ${body.kind} from ${hostname}`
      );
      return Response.json({ ok: true, forwarded: body.kind, delivered });
    }

    return Response.json({ error: "not found" }, { status: 404 });
  },
});

console.error(`[fada-channel] HTTP server listening on 127.0.0.1:${PORT}`);
console.error(
  `[fada-channel] Auth: ${AUTH_TOKEN ? "enabled" : "DISABLED (no token)"}`
);

// ---------------------------------------------------------------------------
// MCP stdio transport (connects to Claude Code)
// ---------------------------------------------------------------------------
const transport = new StdioServerTransport();
await mcp.connect(transport);
