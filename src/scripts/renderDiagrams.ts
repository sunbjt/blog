/**
 * 把文章正文里的 ```mermaid / ```plantuml 代码块渲染成图。
 *
 * 由文章页 src/pages/posts/[...slug]/index.astro 的客户端脚本调用。
 * mermaid 与 plantuml-encoder 都通过动态 import 按需加载——首页、归档、
 * 标签页等不含图表代码块的页面不会加载它们。
 *
 * 前提：astro.config.ts 里 markdown.syntaxHighlight.excludeLangs 包含
 * "mermaid"/"plantuml"，这样这些围栏不会被 Shiki 改写，能保留为
 * <pre><code class="language-…"> 的纯文本结构。
 */

type MermaidApi = {
  initialize: (config: Record<string, unknown>) => void;
  run: (opts: { nodes: HTMLElement[] }) => Promise<void>;
};

/** 已渲染的 mermaid 容器及对应的原始文本（供切换主题时重绘）。 */
const mermaidSources = new WeakMap<HTMLElement, string>();
let mermaidHolders: HTMLElement[] = [];
let renderingMermaid = false;
let themeObserver: MutationObserver | null = null;

function mermaidTheme(): string {
  return document.documentElement.getAttribute("data-theme") === "dark"
    ? "dark"
    : "default";
}

/** 把每个 <pre><code class="language-mermaid"> 原位替换成容器并渲染成 SVG。 */
async function renderMermaid(): Promise<void> {
  if (renderingMermaid) return;

  const codes = Array.from(
    document.querySelectorAll<HTMLElement>("pre > code.language-mermaid")
  );
  if (codes.length === 0) return;

  renderingMermaid = true;
  try {
    const { default: mermaid } = (await import("mermaid")) as unknown as {
      default: MermaidApi;
    };
    mermaid.initialize({
      startOnLoad: false,
      theme: mermaidTheme(),
      securityLevel: "loose",
      fontFamily: "inherit",
    });

    // 每页只持有本页的容器
    mermaidHolders = [];

    for (const code of codes) {
      const pre = code.closest("pre");
      if (!pre) continue;
      const raw = code.textContent ?? "";
      if (!raw.trim()) continue;

      const holder = document.createElement("div");
      holder.className = "mermaid my-6 flex justify-center overflow-x-auto";
      holder.textContent = raw;
      mermaidSources.set(holder, raw);
      mermaidHolders.push(holder);
      pre.replaceWith(holder);

      try {
        await mermaid.run({ nodes: [holder] });
      } catch (err) {
        console.error("[blog] mermaid 渲染失败：", err);
        holder.textContent = mermaidSources.get(holder) ?? raw;
      }
    }

    watchThemeForMermaid();
  } finally {
    renderingMermaid = false;
  }
}

/** 亮/暗主题切换时，用新主题重绘已渲染的 mermaid 图。 */
async function rerenderMermaid(): Promise<void> {
  if (renderingMermaid || mermaidHolders.length === 0) return;

  renderingMermaid = true;
  try {
    const { default: mermaid } = (await import("mermaid")) as unknown as {
      default: MermaidApi;
    };
    mermaid.initialize({
      startOnLoad: false,
      theme: mermaidTheme(),
      securityLevel: "loose",
      fontFamily: "inherit",
    });

    for (const holder of mermaidHolders) {
      holder.textContent = mermaidSources.get(holder) ?? "";
      try {
        await mermaid.run({ nodes: [holder] });
      } catch (err) {
        console.error("[blog] mermaid 重绘失败：", err);
      }
    }
  } finally {
    renderingMermaid = false;
  }
}

function watchThemeForMermaid(): void {
  if (themeObserver) return;
  themeObserver = new MutationObserver(() => {
    void rerenderMermaid();
  });
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });
}

/** 把 <pre><code class="language-plantuml"> 原位替换成远端渲染的 <img>。 */
async function renderPlantuml(): Promise<void> {
  const codes = Array.from(
    document.querySelectorAll<HTMLElement>("pre > code.language-plantuml")
  );
  if (codes.length === 0) return;

  const plantumlEncoder = (await import("plantuml-encoder")).default;
  for (const code of codes) {
    const pre = code.closest("pre");
    if (!pre) continue;
    const raw = (code.textContent ?? "").trim();
    if (!raw) continue;

    const img = document.createElement("img");
    img.src = `https://www.plantuml.com/plantuml/svg/${plantumlEncoder.encode(raw)}`;
    img.alt = "PlantUML diagram";
    img.loading = "lazy";
    img.className = "my-6 max-w-full";
    pre.replaceWith(img);
  }
}

/**
 * 渲染当前页正文里出现的 mermaid / plantuml 图；页面没有图表代码块时是空操作。
 * 可重复调用（图表替换后再次调用不会找到 <pre> 代码块，因此幂等）。
 */
export async function renderDiagrams(): Promise<void> {
  await Promise.all([renderMermaid(), renderPlantuml()]);
}
