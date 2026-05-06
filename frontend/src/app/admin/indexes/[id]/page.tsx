"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useAuth } from "@clerk/nextjs";
import { getIndex, updateIndex } from "@/lib/api";

type IndexEntry = {
  id: string;
  index_name: string;
  project_id: string;
  api_key_env_var: string;
  dimension: number;
  embedding_model: string;
  metric: string;
  domain_description: string;
  sample_queries: string[];
  is_active: boolean;
};

export default function EditIndexPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { getToken } = useAuth();

  const [entry, setEntry] = useState<IndexEntry | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newQuery, setNewQuery] = useState("");

  useEffect(() => {
    async function load() {
      const token = await getToken();
      if (!token) return;
      try {
        const data = await getIndex(token, id);
        setEntry(data);
      } catch (err) {
        setError(String(err));
      }
    }
    load();
  }, [id, getToken]);

  const handleSave = useCallback(async () => {
    if (!entry) return;
    const token = await getToken();
    if (!token) return;
    setSaving(true);
    setError(null);
    try {
      await updateIndex(token, entry.id, {
        domain_description: entry.domain_description,
        sample_queries: entry.sample_queries,
        is_active: entry.is_active,
        embedding_model: entry.embedding_model,
        metric: entry.metric,
      });
      router.push("/admin");
    } catch (err) {
      setError(String(err));
      setSaving(false);
    }
  }, [entry, getToken, router]);

  const addQuery = () => {
    const q = newQuery.trim();
    if (!q || !entry) return;
    setEntry({ ...entry, sample_queries: [...entry.sample_queries, q] });
    setNewQuery("");
  };

  const removeQuery = (i: number) => {
    if (!entry) return;
    setEntry({ ...entry, sample_queries: entry.sample_queries.filter((_, idx) => idx !== i) });
  };

  if (!entry) return <div className="admin-page"><p>{error ?? "Loading…"}</p></div>;

  return (
    <div className="admin-page">
      <div className="admin-header">
        <div>
          <h1>{entry.index_name}</h1>
          <p style={{ color: "#888", margin: "0.25rem 0 0", fontSize: "0.875rem" }}>
            Project {entry.project_id} · {entry.dimension}d · {entry.embedding_model}
          </p>
        </div>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <button className="btn btn-outline" onClick={() => router.push("/admin")}>Cancel</button>
          <button className="btn" onClick={handleSave} disabled={saving}>
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      </div>

      {error && <p style={{ color: "#e05252", marginBottom: "1rem" }}>{error}</p>}

      <div className="edit-form">
        <div className="form-field">
          <label className="form-label">
            Domain Description
            <span style={{ color: "#888", fontWeight: 400, marginLeft: "0.5rem" }}>
              — What topics and content does this index contain? Be specific. The router LLM uses this to decide which index to query.
            </span>
          </label>
          <textarea
            className="form-textarea"
            style={{ minHeight: "140px" }}
            value={entry.domain_description}
            onChange={(e) => setEntry({ ...entry, domain_description: e.target.value })}
            placeholder="e.g. Contains Max's email copywriting frameworks, subject line formulas, and onboarding sequence templates."
          />
        </div>

        <div className="form-field">
          <label className="form-label">Sample Queries</label>
          <p style={{ color: "#888", fontSize: "0.825rem", margin: "0 0 0.5rem" }}>
            Representative questions that should route to this index. Add 3–5 examples.
          </p>
          <div className="tag-list">
            {entry.sample_queries.map((q, i) => (
              <span key={i} className="tag">
                {q}
                <button className="tag-remove" onClick={() => removeQuery(i)}>×</button>
              </span>
            ))}
          </div>
          <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.5rem" }}>
            <input
              className="form-input"
              style={{ flex: 1 }}
              value={newQuery}
              onChange={(e) => setNewQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  addQuery();
                }
              }}
              placeholder="Type a sample query and press Enter"
            />
            <button className="btn btn-sm" onClick={addQuery}>Add</button>
          </div>
        </div>

        <div className="form-field">
          <label className="form-label">Active</label>
          <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={entry.is_active}
              onChange={(e) => setEntry({ ...entry, is_active: e.target.checked })}
            />
            <span style={{ fontSize: "0.9rem" }}>Include this index in query routing</span>
          </label>
        </div>

        <div style={{ display: "flex", gap: "1rem" }}>
          <div className="form-field" style={{ flex: 1 }}>
            <label className="form-label">Embedding Model</label>
            <input
              className="form-input"
              value={entry.embedding_model}
              onChange={(e) => setEntry({ ...entry, embedding_model: e.target.value })}
            />
          </div>
          <div className="form-field" style={{ flex: 1 }}>
            <label className="form-label">Metric</label>
            <select
              className="form-select"
              value={entry.metric}
              onChange={(e) => setEntry({ ...entry, metric: e.target.value })}
            >
              <option value="cosine">cosine</option>
              <option value="dotproduct">dotproduct</option>
              <option value="euclidean">euclidean</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}
