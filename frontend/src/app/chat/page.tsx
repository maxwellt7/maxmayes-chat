"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useAuth } from "@clerk/nextjs";
import { ChatInput } from "@/components/ChatInput";
import { ChatWindow, type Message } from "@/components/ChatWindow";
import { getChatHistory, listChatSessions, startChatStream } from "@/lib/api";

type Session = {
  session_id: string;
  preview: string;
  message_count: number;
  created_at: string;
  last_message_at: string;
};

function formatDate(iso: string): string {
  const d = new Date(iso);
  const now = new Date();
  const diffDays = Math.floor((now.getTime() - d.getTime()) / 86400000);
  if (diffDays === 0) return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return d.toLocaleDateString([], { weekday: "short" });
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

export default function ChatPage() {
  const { getToken } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [streamingResponse, setStreamingResponse] = useState<Response | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const sessionId = useRef<string>(crypto.randomUUID());

  // Load sidebar sessions on mount
  useEffect(() => {
    async function loadSessions() {
      try {
        const token = await getToken();
        if (!token) return;
        const data = await listChatSessions(token);
        setSessions(data.sessions);
      } catch {
        // sidebar sessions are non-critical
      }
    }
    loadSessions();
  }, [getToken]);

  const startNewChat = useCallback(() => {
    sessionId.current = crypto.randomUUID();
    setActiveSessionId(null);
    setMessages([]);
    setError(null);
    setStreamingResponse(null);
    setIsStreaming(false);
  }, []);

  const loadSession = useCallback(async (sid: string) => {
    try {
      const token = await getToken();
      if (!token) return;
      const history = await getChatHistory(token, sid);
      setMessages(
        history.messages.map((m, i) => ({
          id: `hist-${sid}-${i}`,
          role: m.role as "user" | "assistant",
          content: m.content,
        }))
      );
      sessionId.current = sid;
      setActiveSessionId(sid);
      setError(null);
    } catch {
      setError("Could not load that session.");
    }
  }, [getToken]);

  const handleSend = useCallback(async (message: string) => {
    const token = await getToken();
    if (!token) return;

    setError(null);
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "user", content: message },
    ]);
    setIsStreaming(true);

    try {
      const response = await startChatStream(token, message, sessionId.current);
      setStreamingResponse(response);
    } catch (err) {
      setIsStreaming(false);
      setError(String(err));
    }
  }, [getToken]);

  const handleStreamComplete = useCallback(async (text: string) => {
    setStreamingResponse(null);
    setIsStreaming(false);
    const completedSessionId = sessionId.current;
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "assistant", content: text },
    ]);
    // Refresh sidebar
    try {
      const token = await getToken();
      if (!token) return;
      const data = await listChatSessions(token);
      setSessions(data.sessions);
      setActiveSessionId(completedSessionId);
    } catch { /* non-critical */ }
  }, [getToken]);

  const handleStreamError = useCallback((err: string) => {
    setStreamingResponse(null);
    setIsStreaming(false);
    setError(err);
  }, []);

  const isActive = (sid: string) => sid === activeSessionId;

  return (
    <div className="chat-layout">
      {/* Sidebar */}
      <aside className="chat-sidebar">
        <div className="chat-sidebar-header">
          <span className="chat-sidebar-title">Inquiries</span>
          <button className="btn btn-sm btn-outline" onClick={startNewChat}>
            + New
          </button>
        </div>
        <div className="chat-sidebar-list">
          {sessions.length === 0 && (
            <p className="chat-sidebar-empty">No prior inquiries.</p>
          )}
          {sessions.map((s) => (
            <button
              key={s.session_id}
              className={`chat-session-item${isActive(s.session_id) ? " chat-session-active" : ""}`}
              onClick={() => loadSession(s.session_id)}
            >
              <span className="chat-session-preview">
                {s.preview || "—"}
              </span>
              <span className="chat-session-meta">
                {formatDate(s.last_message_at)}
              </span>
            </button>
          ))}
        </div>
      </aside>

      {/* Main chat area */}
      <div className="chat-main">
        <div className="chat-masthead">
          <div className="chat-masthead-title">
            {activeSessionId ? "Inquiry" : "Today’s Inquiry"}
          </div>
          <div className="chat-masthead-meta">
            {messages.length > 0 ? `${messages.length} entries` : "blank page"}
          </div>
        </div>

        {messages.length === 0 && !isStreaming ? (
          <div className="chat-welcome">
            <p className="chat-welcome-mark">¶</p>
            <p className="chat-welcome-text">
              Ask anything from your knowledge base.
            </p>
            <p className="chat-welcome-hint">
              Copywriting, marketing strategy, personal notes, fitness —
              whatever you&apos;ve indexed.
            </p>
          </div>
        ) : (
          <ChatWindow
            messages={messages}
            streamingResponse={streamingResponse}
            onStreamComplete={handleStreamComplete}
            onStreamError={handleStreamError}
          />
        )}

        {error && <p className="chat-error">{error}</p>}
        <ChatInput onSend={handleSend} disabled={isStreaming} />
      </div>
    </div>
  );
}
