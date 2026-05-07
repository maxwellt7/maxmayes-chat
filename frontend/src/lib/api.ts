const API_URL = process.env.NEXT_PUBLIC_API_URL;

if (!API_URL) {
  console.warn("NEXT_PUBLIC_API_URL is not set.");
}

function apiUrl(path: string): string {
  if (!API_URL) throw new Error("Missing NEXT_PUBLIC_API_URL");
  return `${API_URL}${path}`;
}

// ---- Chat ----

export async function getChatHistory(token: string, sessionId: string) {
  const res = await fetch(apiUrl(`/api/chat/history/${sessionId}`), {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error(`Failed to fetch history: ${res.status}`);
  return res.json() as Promise<{ messages: { role: string; content: string; created_at: string }[] }>;
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
  if (!res.ok) throw new Error(`Chat request failed: ${res.status}`);
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
