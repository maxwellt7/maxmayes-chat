"use client";

import { useEffect, useRef } from "react";
import { MessageBubble } from "./MessageBubble";
import { StreamingResponse } from "./StreamingResponse";

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

type Props = {
  messages: Message[];
  streamingResponse: Response | null;
  onStreamComplete: (text: string) => void;
  onStreamError: (error: string) => void;
};

export function ChatWindow({ messages, streamingResponse, onStreamComplete, onStreamError }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingResponse]);

  return (
    <div className="chat-window">
      {messages.map((m) => (
        <MessageBubble key={m.id} role={m.role} content={m.content} />
      ))}
      {streamingResponse && (
        <StreamingResponse
          response={streamingResponse}
          onComplete={onStreamComplete}
          onError={onStreamError}
        />
      )}
      <div ref={bottomRef} />
    </div>
  );
}
