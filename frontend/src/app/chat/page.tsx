"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useAuth } from "@clerk/nextjs";
import { ChatInput } from "@/components/ChatInput";
import { ChatWindow, type Message } from "@/components/ChatWindow";
import { getChatHistory, startChatStream } from "@/lib/api";

const SESSION_KEY = "pinecone-chat-session";

function getOrCreateSessionId(): string {
  if (typeof window === "undefined") return crypto.randomUUID();
  const existing = localStorage.getItem(SESSION_KEY);
  if (existing) return existing;
  const id = crypto.randomUUID();
  localStorage.setItem(SESSION_KEY, id);
  return id;
}

export default function ChatPage() {
  const { getToken } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [streamingResponse, setStreamingResponse] = useState<Response | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionId = useRef(getOrCreateSessionId());

  useEffect(() => {
    async function loadHistory() {
      try {
        const token = await getToken();
        if (!token) return;
        const history = await getChatHistory(token, sessionId.current);
        setMessages(
          history.messages.map((m, i) => ({
            id: `hist-${i}`,
            role: m.role as "user" | "assistant",
            content: m.content,
          }))
        );
      } catch {
        // no history yet — start fresh
      }
    }
    loadHistory();
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

  const handleStreamComplete = useCallback((text: string) => {
    setStreamingResponse(null);
    setIsStreaming(false);
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "assistant", content: text },
    ]);
  }, []);

  const handleStreamError = useCallback((err: string) => {
    setStreamingResponse(null);
    setIsStreaming(false);
    setError(err);
  }, []);

  return (
    <div className="chat-page">
      <div className="chat-masthead">
        <div className="chat-masthead-title">Today&apos;s Inquiry</div>
        <div className="chat-masthead-meta">
          {messages.length > 0 ? `${messages.length} entries` : "blank page"}
        </div>
      </div>
      <ChatWindow
        messages={messages}
        streamingResponse={streamingResponse}
        onStreamComplete={handleStreamComplete}
        onStreamError={handleStreamError}
      />
      {error && <p className="chat-error">{error}</p>}
      <ChatInput onSend={handleSend} disabled={isStreaming} />
    </div>
  );
}
