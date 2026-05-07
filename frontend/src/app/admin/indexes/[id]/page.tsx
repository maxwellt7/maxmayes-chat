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

  if (!entry) return <div className="edit-page"><p className="loading">{error ?? "Drawing the page…"}</p></div>;

  return (
    <div className="edit-page">
      <div className="edit-header">
        <div className="edit-header-info">
          <h1>{entry.index_name}</h1>
          <div className="edit-header-meta">
            Project {entry.project_id} · {entry.dimension}d · {entry.embedding_model}
          </div>
        </div>
        <div className="edit-header-actions">
          <button className="btn btn-outline btn-sm" onClick={() => router.push("/admin")}>Cancel</button>
          <button className="btn btn-sm" onClick={handleSave} disabled={saving}>
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      </div>

      {error && <p className="error-banner">{error}</p>}

      <div className="edit-form">
        <div className="form-field">
          <label className="form-label">Domain Description</label>
          <span className="form-label-aside">
            What lives in this index? Be specific — content, frameworks, projects. The router uses this to decide which index a question belongs to.
          </span>
          <textarea
            className="form-textarea"
            style={{ minHeight: "160px" }}
            value={entry.domain_description}
            onChange={(e) => setEntry({ ...entry, domain_description: e.target.value })}
            placeholder="e.g. Email copywriting frameworks, subject line formulas, and onboarding sequence templates from 2022–present."
          />
        </div>

        <div className="form-field">
          <label className="form-label">Sample Queries</label>
          <span className="form-label-aside">
            Three to five questions that should land here. Phrasing matters — write them how you&apos;d actually ask.
          </span>
          <div className="tag-list">
            {entry.sample_queries.map((q, i) => (
              <span key={i} className="tag">
                {q}
                <button className="tag-remove" onClick={() => removeQuery(i)}>×</button>
              </span>
            ))}
          </div>
          <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.4rem" }}>
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
              placeholder="A representative question…"
            />
            <button className="btn btn-sm btn-outline" onClick={addQuery}>Add</button>
          </div>
        </div>

        <div className="form-field">
          <label className="form-label">Inclusion</label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={entry.is_active}
              onChange={(e) => setEntry({ ...entry, is_active: e.target.checked })}
            />
            <span className="checkbox-label">Include this index in routing decisions</span>
          </label>
        </div>

        <div style={{ display: "flex", gap: "1.5rem" }}>
          <div className="form-field" style={{ flex: 1 }}>
            <label className="form-label">Embedding Model</label>
            <input className="form-input" value={entry.embedding_model} onChange={(e) => setEntry({ ...entry, embedding_model: e.target.value })} />
          </div>
          <div className="form-field" style={{ flex: 1 }}>
            <label className="form-label">Metric</label>
            <select className="form-select" value={entry.metric} onChange={(e) => setEntry({ ...entry, metric: e.target.value })}>
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
