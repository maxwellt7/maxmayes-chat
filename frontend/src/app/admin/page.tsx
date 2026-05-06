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
        <h1>Index Registry</h1>
        <button className="btn" onClick={handleDiscover} disabled={discovering}>
          {discovering ? "Scanning Pinecone…" : "Discover from Pinecone"}
        </button>
      </div>

      {error && <p style={{ color: "#e05252", marginBottom: "1rem" }}>{error}</p>}

      {discovered && (
        <div className="card" style={{ marginBottom: "1.5rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
            <h3 style={{ margin: 0 }}>Discovered Indexes ({discovered.length})</h3>
            <button className="btn btn-sm btn-outline" onClick={() => setDiscovered(null)}>Dismiss</button>
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
                    <td>{d.index_name}</td>
                    <td>{d.project_id}</td>
                    <td>{d.dimension}</td>
                    <td>{d.metric}</td>
                    <td>
                      {d.already_in_registry ? (
                        <span style={{ color: "#888", fontSize: "0.8rem" }}>Already imported</span>
                      ) : (
                        <button
                          className="btn btn-sm"
                          disabled={importing.has(key)}
                          onClick={() => handleImport(d)}
                        >
                          {importing.has(key) ? "Importing…" : "Import"}
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
        <p>Loading registry…</p>
      ) : (
        <table className="registry-table">
          <thead>
            <tr><th>Index</th><th>Project</th><th>Dim</th><th>Model</th><th>Status</th><th>Description</th><th></th></tr>
          </thead>
          <tbody>
            {indexes.length === 0 && (
              <tr><td colSpan={7} style={{ color: "#888", padding: "2rem 0" }}>No indexes yet. Click &quot;Discover from Pinecone&quot; to get started.</td></tr>
            )}
            {indexes.map((idx) => (
              <tr key={idx.id}>
                <td>{idx.index_name}</td>
                <td>{idx.project_id}</td>
                <td>{idx.dimension}</td>
                <td style={{ fontSize: "0.8rem", color: "#aaa" }}>{idx.embedding_model}</td>
                <td>
                  <button
                    className={`badge ${idx.is_active ? "badge-active" : "badge-inactive"}`}
                    onClick={() => handleToggleActive(idx)}
                    style={{ cursor: "pointer", background: "none" }}
                  >
                    {idx.is_active ? "Active" : "Inactive"}
                  </button>
                </td>
                <td style={{ maxWidth: "240px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "#aaa", fontSize: "0.85rem" }}>
                  {idx.domain_description || <em style={{ color: "#555" }}>No description yet</em>}
                </td>
                <td>
                  <Link href={`/admin/indexes/${idx.id}`} className="btn btn-sm btn-outline">Edit</Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
