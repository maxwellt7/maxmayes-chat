const API_URL = process.env.NEXT_PUBLIC_API_URL;

if (!API_URL) {
  console.warn("NEXT_PUBLIC_API_URL is not set.");
}

function apiUrl(path: string): string {
  if (!API_URL) throw new Error("Missing NEXT_PUBLIC_API_URL");
  return `${API_URL}${path}`;
}

// ---- Chat ----

export async function listChatSessions(token: string) {
  const res = await fetch(apiUrl("/api/chat/sessions"), {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error(`Failed to fetch sessions: ${res.status}`);
  return res.json() as Promise<{
    sessions: {
      session_id: string;
      preview: string;
      message_count: number;
      created_at: string;
      last_message_at: string;
    }[];
  }>;
}

export async function getChatHistory(token: string, sessionId: string) {
  const res = await fetch(apiUrl(`/api/chat/history/${sessionId}`), {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error(`Failed to fetch history: ${res.status}`);
  return res.json() as Promise<{ messages: { role: string; content: string; created_at: string }[] }>;
}

/** Maximum accepted by the server; mirrors `MAX_CHAT_MESSAGE_CHARS`. */
export const MAX_CHAT_MESSAGE_CHARS = 4000;

/** Pull the server's client-safe message out of an error response.
 *
 *  Rate limits and the daily spend ceiling are expected outcomes with something
 *  useful to say, so surfacing "Chat request failed: 429" instead would be
 *  throwing away the only actionable part. Falls back to the status code when the
 *  body is not in the taxonomy's shape. */
async function errorMessage(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json();
    if (body?.error?.message) return body.error.message as string;
  } catch {
    // no JSON body — fall through
  }
  return `${fallback}: ${res.status}`;
}

export async function startChatStream(
  token: string,
  message: string,
  sessionId: string
): Promise<Response> {
  const res = await fetch(apiUrl("/api/chat"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ message, session_id: sessionId }),
  });
  if (!res.ok) {
    throw new Error(await errorMessage(res, "Chat request failed"));
  }
  return res;
}

// ---- Admin ----

export async function listIndexes(token: string) {
  const res = await fetch(apiUrl("/api/admin/indexes?include_inactive=true"), {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error(`Failed to list indexes: ${res.status}`);
  return res.json();
}

export async function getIndex(token: string, id: string) {
  const res = await fetch(apiUrl(`/api/admin/indexes/${id}`), {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error(`Failed to get index: ${res.status}`);
  return res.json();
}

export async function updateIndex(token: string, id: string, data: Record<string, unknown>) {
  const res = await fetch(apiUrl(`/api/admin/indexes/${id}`), {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`Failed to update index: ${res.status}`);
  return res.json();
}

export async function createIndex(token: string, data: Record<string, unknown>) {
  const res = await fetch(apiUrl("/api/admin/indexes"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`Failed to create index: ${res.status}`);
  return res.json();
}

export async function discoverIndexes(token: string) {
  const res = await fetch(apiUrl("/api/admin/discover"), {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error(`Discovery failed: ${res.status}`);
  return res.json() as Promise<{
    discovered: { index_name: string; project_id: string; dimension: number; metric: string; already_in_registry: boolean }[];
    count: number;
    partial_failures?: string[];
  }>;
}

export async function importIndex(
  token: string,
  discovered: { index_name: string; project_id: string; dimension: number; metric: string }
) {
  const dimensionModelMap: Record<number, string> = {
    1024: "embed-english-v3.0",
    1536: "text-embedding-3-small",
    2048: "text-embedding-3-large",
  };
  return createIndex(token, {
    index_name: discovered.index_name,
    project_id: discovered.project_id,
    api_key_env_var: `PINECONE_API_KEY_${discovered.project_id}`,
    dimension: discovered.dimension,
    embedding_model: dimensionModelMap[discovered.dimension] ?? "text-embedding-3-small",
    metric: discovered.metric,
    domain_description: "",
    sample_queries: [],
    is_active: false,
  });
}

// ---- Phase 0 Audit ----

export type Disposition = "KEEP" | "MERGE" | "RE-INGEST" | "ARCHIVE" | "SPLIT";

export interface AuditRow {
  id: string;
  index_name: string;
  project_id: string;
  record_count: number;
  embedding_model: string | null;
  dominant_domain: string | null;
  topic_tags: string[];
  sample_chunks: Array<{ text: string; metadata: Record<string, unknown> }>;
  proposed_disposition: Disposition;
  proposed_target_index: string | null;
  approved_disposition: string | null;
}

export async function fetchLatestAudit(token: string): Promise<AuditRow[]> {
  const res = await fetch(apiUrl("/api/admin/audit/latest"), {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error(`audit fetch failed: ${res.status}`);
  return res.json();
}

export async function kickOffAudit(
  token: string,
  mode: "dry_run" | "execute"
): Promise<{ audit_id: string; row_count: number; audit_date: string }> {
  const res = await fetch(apiUrl("/api/admin/audit"), {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  if (!res.ok) throw new Error(`audit kickoff failed: ${res.status}`);
  return res.json();
}

export async function approveDispositions(
  token: string,
  auditDate: string,
  updates: Array<{
    audit_row_id: string;
    approved_disposition: Disposition;
    approved_target_index?: string | null;
  }>
): Promise<{ created_jobs: string[]; updated_rows: number }> {
  const res = await fetch(
    apiUrl(`/api/admin/audit/${auditDate}/dispositions`),
    {
      method: "POST",
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: JSON.stringify(updates),
    }
  );
  if (!res.ok) throw new Error(`dispositions failed: ${res.status}`);
  return res.json();
}

export async function triggerIngest(token: string, jobId: string): Promise<void> {
  const res = await fetch(apiUrl(`/api/admin/ingest/${jobId}/run`), {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error(`ingest trigger failed: ${res.status}`);
}
