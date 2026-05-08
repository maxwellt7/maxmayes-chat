import { readFile } from "node:fs/promises";
import path from "node:path";

async function loadSpecMarkup(): Promise<string> {
  const specFilePath = path.join(process.cwd(), "..", "frontend-design.html");
  try {
    return await readFile(specFilePath, "utf8");
  } catch {
    return "<html><body><p>frontend-design.html not found.</p></body></html>";
  }
}

export default async function SpecsPage() {
  const markup = await loadSpecMarkup();

  return (
    <main style={{ maxWidth: "100%", padding: 0 }}>
      <iframe
        title="Frontend Design Spec"
        srcDoc={markup}
        style={{ width: "100%", height: "100vh", border: "none" }}
      />
    </main>
  );
}
