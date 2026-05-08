"use client";

import ReactMarkdown from "react-markdown";
import remarkBreaks from "remark-breaks";
import remarkGfm from "remark-gfm";

type Props = {
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
};

const ROLE_LABEL: Record<Props["role"], string> = {
  user: "You",
  assistant: "Max",
};

export function MessageBubble({ role, content, isStreaming }: Props) {
  return (
    <div className={`message-bubble message-${role}`}>
      <div className="label">{ROLE_LABEL[role]}</div>
      <div className="bubble-content">
        <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
          {content}
        </ReactMarkdown>
        {isStreaming && <span className="streaming-cursor" />}
      </div>
    </div>
  );
}
