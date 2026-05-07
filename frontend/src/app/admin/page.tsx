"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useAuth } from "@clerk/nextjs";
import { discoverIndexes, importIndex, listIndexes, updateIndex } from "@/lib/api";

type IndexEntry = {
  id: string;
  index_name: string;
  project_id: string;
  dimension: number;
  embedding_model: string;
  is_active: boolean;
  domain_description: string;
};

type DiscoveredEntry = {
  index_name: string;
  project_id: string;
  dimension: number;
  metric: string;
  already_in_registry: boolean;
};

export default function AdminPage() {
  const { getToken } = useAuth();
  const [indexes, setIndexes] = useState<IndexEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [discovering, setDiscovering] = useState(false);
  const [discovered, setDiscovered] = useState<DiscoveredEntry[] | null>(null);
  const [importing, setImporting] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  const loadIndexes = useCallback(async () => {
    const token = await getToken();
    if (!token) return;
    try {
      const data = await listIndexes(token);
      setIndexes(data.indexes);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, [getToken]);

  useEffect(() => { loadIndexes(); }, [loadIndexes]);

  const handleDiscover = async () => {
    const token = await getToken();
    if (!token) return;
    setDiscovering(true);
    setError(null);
    try {
      const data = await discoverIndexes(token);
      setDiscovered(data.discovered);
    } catch (err) {
      setError(String(err));
    } finally {
      setDiscovering(false);
    }
  };

  const handleImport = async (entry: DiscoveredEntry) => {
    const token = await getToken();
    if (!token) return;
    const key = `${entry.index_name}-${entry.project_id}`;
    setImporting((prev) => new Set(prev).add(key));
    try {
      await importIndex(token, entry);
      await loadIndexes();
    } catch (err) {
      setError(String(err));
    } finally {
      setImporting((prev) => { const s = new Set(prev); s.delete(key); return s; });
    }
  };

  const handleToggleActive = async (idx: IndexEntry) => {
    const token = await getToken();
    if (!token) return;
    try {
      await updateIndex(token, idx.id, { is_active: !idx.is_active });
      await loadIndexes();
    } catch (err) {
      setError(String(err));
    }
  };

  return (
    <div className="admin-page">
      <div className="admin-header">
        <div>
          <h1>The Catalogue</h1>
          <p className="subtitle">Forty-odd indices, indexed and laid out for inspection.</p>
        </div>
        <button className="btn" onClick={handleDiscover} disabled={discovering}>
          {discovering ? "Surveying…" : "Discover from Pinecone"}
        </button>
      </div>

      {error && <p className="error-banner">{error}</p>}

      {discovered && (
        <div className="discovered-panel">
          <div className="discovered-panel-header">
            <h3>
              Found in Pinecone
              <span style={{ color: "var(--ink-faint)", fontFamily: "var(--font-mono)", fontSize: "0.78rem", marginLeft: "0.5rem", letterSpacing: "0.1em" }}>
                {discovered.length} entries
              </span>
            </h3>
            <button className="btn btn-sm btn-ghost" onClick={() => setDiscovered(null)}>Dismiss</button>
          </div>
          <table className="registry-table">
            <thead>
              <tr><th>Index</th><th>Project</th><th>Dimension</th><th>Metric</th><th></th></tr>
            </thead>
            <tbody>
              {discovered.map((d) => {
                const key = `${d.index_name}-${d.project_id}`;
                return (
                  <tr key={key}>
                    <td><span className="registry-name">{d.index_name}</span></td>
                    <td><span className="registry-meta">P{d.project_id}</span></td>
                    <td><span className="registry-meta">{d.dimension}</span></td>
                    <td><span className="registry-meta">{d.metric}</span></td>
                    <td>
                      {d.already_in_registry ? (
                        <span className="label">In Catalogue</span>
                      ) : (
                        <button className="btn btn-sm btn-outline" disabled={importing.has(key)} onClick={() => handleImport(d)}>
                          {importing.has(key) ? "Adding…" : "Add"}
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {loading ? (
        <p className="loading">Drawing the catalogue…</p>
      ) : (
        <table className="registry-table">
          <thead>
            <tr>
              <th>Index</th>
              <th>Project</th>
              <th>Dim</th>
              <th>Status</th>
              <th>Description</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {indexes.length === 0 && (
              <tr><td colSpan={6} className="empty-row">The catalogue is empty. Click &ldquo;Discover from Pinecone&rdquo; to seed it.</td></tr>
            )}
            {indexes.map((idx) => (
              <tr key={idx.id}>
                <td><span className="registry-name">{idx.index_name}</span></td>
                <td><span className="registry-meta">P{idx.project_id}</span></td>
                <td><span className="registry-meta">{idx.dimension}</span></td>
                <td>
                  <button
                    className={`sigil ${idx.is_active ? "sigil-active" : ""}`}
                    onClick={() => handleToggleActive(idx)}
                    aria-label={idx.is_active ? "Active — click to deactivate" : "Inactive — click to activate"}
                  >
                    <span className="sigil-mark" />
                    {idx.is_active ? "Open" : "Closed"}
                  </button>
                </td>
                <td>
                  <div className="registry-description">
                    {idx.domain_description ? idx.domain_description : <em>not yet described</em>}
                  </div>
                </td>
                <td>
                  <Link href={`/admin/indexes/${idx.id}`} className="btn btn-sm btn-ghost">Edit →</Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
