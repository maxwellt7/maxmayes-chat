"use client";

type Props = {
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
};

export function MessageBubble({ role, content, isStreaming }: Props) {
  return (
    <div className={`message-bubble message-${role}`}>
      <div className="bubble-content">
        {content}
        {isStreaming && <span className="streaming-cursor" />}
      </div>
    </div>
  );
}
