import Link from "next/link";
import { auth } from "@clerk/nextjs/server";

export default async function HomePage() {
  const { userId } = await auth();

  return (
    <main className="landing">
      <div className="landing-figure" aria-hidden="true">¶</div>
      <div className="landing-inner">
        <p className="landing-prelude">Maxwell Mayes · Private Codex</p>
        <h1 className="landing-title">
          The <em>Archive</em>,<br />
          rendered in voice.
        </h1>
        <p className="landing-lede">
          A query layer over my private knowledge — forty-three indices spanning
          copywriting, frameworks, conversations, and uncatalogued thought. Ask
          something specific. The answer arrives in my voice, drawn entirely
          from what I have written.
        </p>
        <div className="landing-actions">
          {userId ? (
            <>
              <Link href="/chat" className="btn">Begin inquiry</Link>
              <Link href="/admin" className="btn btn-outline">Manage indices</Link>
            </>
          ) : (
            <>
              <Link href="/sign-in" className="btn">Enter</Link>
              <span className="label" style={{ color: "var(--ink-faint)" }}>· access by invitation</span>
            </>
          )}
        </div>
        <div className="landing-stats">
          <div>
            <div className="landing-stat-num">43</div>
            <div className="landing-stat-label">Active Indices</div>
          </div>
          <div>
            <div className="landing-stat-num">3</div>
            <div className="landing-stat-label">Pinecone Projects</div>
          </div>
          <div>
            <div className="landing-stat-num">5</div>
            <div className="landing-stat-label">Pipeline Stages</div>
          </div>
        </div>
      </div>
    </main>
  );
}
