"use client";

import { useAuth } from "@clerk/nextjs";
import { useCallback, useEffect, useState } from "react";

import {
  approveDispositions,
  AuditRow,
  Disposition,
  fetchLatestAudit,
  kickOffAudit,
  triggerIngest,
} from "@/lib/api";

const DISPOSITIONS: Disposition[] = ["KEEP", "MERGE", "RE-INGEST", "ARCHIVE", "SPLIT"];

const TARGET_INDEXES = [
  "max-strategy",
  "max-marketing",
  "max-voice",
  "max-personal",
  "max-tools-ops",
  "max-content-library",
  "max-verticals",
];

interface Override {
  disposition?: Disposition;
  target?: string | null;
}

export default function AuditPage() {
  const { getToken } = useAuth();
  const [rows, setRows] = useState<AuditRow[]>([]);
  const [overrides, setOverrides] = useState<Record<string, Override>>({});
  const [auditRunning, setAuditRunning] = useState(false);
  const [approving, setApproving] = useState(false);
  const [jobIds, setJobIds] = useState<string[]>([]);
  const [status, setStatus] = useState<string>("");

  const load = useCallback(async () => {
    const token = await getToken();
    if (!token) return;
    try {
      const data = await fetchLatestAudit(token);
      setRows(data);
    } catch (err) {
      setStatus(`Failed to load audit: ${(err as Error).message}`);
    }
  }, [getToken]);

  useEffect(() => {
    load();
  }, [load]);

  const runAudit = async () => {
    setAuditRunning(true);
    setStatus("Running audit — this may take a minute...");
    try {
      const token = await getToken();
      if (!token) return;
      const result = await kickOffAudit(token, "execute");
      setStatus(`Audit complete: ${result.row_count} rows`);
      await load();
    } catch (err) {
      setStatus(`Audit failed: ${(err as Error).message}`);
    } finally {
      setAuditRunning(false);
    }
  };

  const approveAll = async () => {
    if (rows.length === 0) {
      setStatus("No audit rows to approve. Run audit first.");
      return;
    }
    setApproving(true);
    setStatus("Approving dispositions...");
    try {
      const token = await getToken();
      if (!token) return;
      const auditDate = new Date().toISOString().slice(0, 10);
      const updates = rows.map((r) => {
        const ovr = overrides[r.id];
        const disposition = ovr?.disposition ?? r.proposed_disposition;
        const target = ovr?.target ?? r.proposed_target_index ?? undefined;
        return {
          audit_row_id: r.id,
          approved_disposition: disposition,
          approved_target_index: target ?? null,
        };
      });
      const result = await approveDispositions(token, auditDate, updates);
      setJobIds(result.created_jobs);
      setStatus(
        `Approved ${result.updated_rows} rows, created ${result.created_jobs.length} ingest jobs`
      );
    } catch (err) {
      setStatus(`Approve failed: ${(err as Error).message}`);
    } finally {
      setApproving(false);
    }
  };

  const startAllIngest = async () => {
    if (jobIds.length === 0) {
      setStatus("No jobs to run. Approve dispositions first.");
      return;
    }
    setStatus(`Starting ${jobIds.length} ingest jobs...`);
    const token = await getToken();
    if (!token) return;
    let started = 0;
    for (const jid of jobIds) {
      try {
        await triggerIngest(token, jid);
        started += 1;
      } catch (err) {
        console.error("ingest trigger failed", jid, err);
      }
    }
    setStatus(`Started ${started}/${jobIds.length} ingest jobs (running in background)`);
  };

  return (
    <main
      style={{
        padding: "2rem",
        fontFamily: "var(--font-mono, ui-monospace, monospace)",
        maxWidth: "1400px",
        margin: "0 auto",
      }}
    >
      <h1>Phase 0 — Index Audit</h1>
      <p style={{ color: "#666", marginBottom: "1.5rem" }}>
        Run audit → review each row → adjust dispositions → approve all → start ingest jobs.
      </p>

      <div style={{ marginBottom: "1rem", display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
        <button onClick={runAudit} disabled={auditRunning}>
          {auditRunning ? "Running..." : "Run audit"}
        </button>
        <button onClick={approveAll} disabled={approving || rows.length === 0}>
          {approving ? "Approving..." : "Approve all (with overrides)"}
        </button>
        <button onClick={startAllIngest} disabled={jobIds.length === 0}>
          Start all ingest jobs ({jobIds.length})
        </button>
      </div>

      {status && (
        <div
          style={{
            padding: "0.5rem 0.75rem",
            background: "#f4f1ea",
            border: "1px solid #d5cfc0",
            marginBottom: "1rem",
          }}
        >
          {status}
        </div>
      )}

      {rows.length === 0 ? (
        <p style={{ color: "#888" }}>No audit rows yet. Click &quot;Run audit&quot;.</p>
      ) : (
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontSize: "0.92rem",
          }}
        >
          <thead>
            <tr style={{ background: "#f4f1ea" }}>
              <th align="left" style={{ padding: "0.4rem" }}>
                Index
              </th>
              <th align="left" style={{ padding: "0.4rem" }}>
                Records
              </th>
              <th align="left" style={{ padding: "0.4rem" }}>
                Domain
              </th>
              <th align="left" style={{ padding: "0.4rem" }}>
                Proposed
              </th>
              <th align="left" style={{ padding: "0.4rem" }}>
                Override
              </th>
              <th align="left" style={{ padding: "0.4rem" }}>
                Target index
              </th>
              <th align="left" style={{ padding: "0.4rem" }}>
                Sample
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} style={{ borderTop: "1px solid #ddd" }}>
                <td style={{ padding: "0.4rem", fontWeight: 600 }}>{r.index_name}</td>
                <td style={{ padding: "0.4rem" }}>{r.record_count.toLocaleString()}</td>
                <td style={{ padding: "0.4rem" }}>{r.dominant_domain || "-"}</td>
                <td style={{ padding: "0.4rem" }}>{r.proposed_disposition}</td>
                <td style={{ padding: "0.4rem" }}>
                  <select
                    value={overrides[r.id]?.disposition ?? r.proposed_disposition}
                    onChange={(e) =>
                      setOverrides((o) => ({
                        ...o,
                        [r.id]: {
                          ...o[r.id],
                          disposition: e.target.value as Disposition,
                        },
                      }))
                    }
                  >
                    {DISPOSITIONS.map((d) => (
                      <option key={d} value={d}>
                        {d}
                      </option>
                    ))}
                  </select>
                </td>
                <td style={{ padding: "0.4rem" }}>
                  <input
                    list={`target-${r.id}`}
                    defaultValue={r.proposed_target_index || ""}
                    onChange={(e) =>
                      setOverrides((o) => ({
                        ...o,
                        [r.id]: {
                          ...o[r.id],
                          disposition: o[r.id]?.disposition ?? r.proposed_disposition,
                          target: e.target.value || null,
                        },
                      }))
                    }
                    style={{ width: "180px" }}
                  />
                  <datalist id={`target-${r.id}`}>
                    {TARGET_INDEXES.map((t) => (
                      <option key={t} value={t} />
                    ))}
                  </datalist>
                </td>
                <td style={{ padding: "0.4rem" }}>
                  <details>
                    <summary style={{ cursor: "pointer" }}>
                      view {r.sample_chunks.length}
                    </summary>
                    <pre
                      style={{
                        maxWidth: "700px",
                        whiteSpace: "pre-wrap",
                        background: "#fafaf7",
                        padding: "0.5rem",
                        fontSize: "0.82rem",
                      }}
                    >
                      {r.sample_chunks.map((c) => c.text).join("\n\n---\n\n")}
                    </pre>
                  </details>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </main>
  );
}
