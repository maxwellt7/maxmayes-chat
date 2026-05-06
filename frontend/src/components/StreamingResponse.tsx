"use client";

import { useEffect, useRef, useState } from "react";
import { readSSEStream } from "@/lib/sse";
import { MessageBubble } from "./MessageBubble";

type Props = {
  response: Response;
  onComplete: (fullText: string) => void;
  onError: (error: string) => void;
};

export function StreamingResponse({ response, onComplete, onError }: Props) {
  const [text, setText] = useState("");
  const fullTextRef = useRef("");

  useEffect(() => {
    let cancelled = false;

    async function stream() {
      try {
        for await (const event of readSSEStream(response)) {
          if (cancelled) return;
          if ("error" in event) {
            onError(event.error);
            return;
          }
          fullTextRef.current += event.token;
          setText(fullTextRef.current);
        }
        if (!cancelled) onComplete(fullTextRef.current);
      } catch (err) {
        if (!cancelled) onError(String(err));
      }
    }

    stream();
    return () => { cancelled = true; };
  }, [response]);

  return <MessageBubble role="assistant" content={text} isStreaming />;
}
