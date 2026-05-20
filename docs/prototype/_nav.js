/* private-rag · prototype · shared top-nav.
 *
 * Single source of truth for the cross-page nav bar. Each HTML page
 * mounts the nav with:
 *
 *   <div id="topnav-mount"></div>
 *   <script src="_nav.js" defer></script>
 *
 * Generation is a hover/click dropdown that lists both LLM paths.
 * The script auto-detects the current page from `location.pathname`
 * and applies `class="current"` to the matching link (also opens
 * the Generation group when the current page is a generation child).
 */

(function () {
  const ITEMS = [
    { href: "storage.html", label: "Storage — CAPRISE" },
    { href: "embedding.html", label: "Embedding — GELO" },
    { href: "reranking.html", label: "Reranking — GELO" },
    {
      label: "Generation",
      children: [
        { href: "gelo-llm.html", label: "GELO LLM" },
        { href: "aloepri-llm.html", label: "AloePri LLM" },
      ],
    },
    { href: "graphrag.html", label: "GraphRAG — Compass" },
  ];

  function currentPage() {
    const p = location.pathname.split("/").pop();
    return p && p.length ? p : "index.html";
  }

  function escapeHTML(s) {
    return s
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function renderItem(item, here) {
    if (item.children) {
      const childHrefs = item.children.map((c) => c.href);
      const groupOpen = childHrefs.includes(here);
      const groupClass = groupOpen ? "navgroup open" : "navgroup";
      const childLinks = item.children
        .map((c) => {
          const cur = c.href === here ? " current" : "";
          return `<a class="${cur.trim()}" href="${escapeHTML(c.href)}">${escapeHTML(c.label)}</a>`;
        })
        .join("");
      const labelCls = groupOpen ? "navgroup-label current" : "navgroup-label";
      return `<div class="${groupClass}"><span class="${labelCls}" tabindex="0">${escapeHTML(item.label)} ▾</span><div class="navgroup-menu">${childLinks}</div></div>`;
    }
    const cur = item.href === here ? " current" : "";
    return `<a class="${cur.trim()}" href="${escapeHTML(item.href)}">${escapeHTML(item.label)}</a>`;
  }

  function render() {
    const here = currentPage();
    const links = ITEMS.map((i) => renderItem(i, here)).join("");
    const html = `
      <nav class="topnav">
        <a class="brand" href="index.html">
          private-rag
        </a>
        <div class="links">${links}</div>
      </nav>`;

    const mount = document.getElementById("topnav-mount");
    if (mount) {
      mount.outerHTML = html;
    } else {
      const wrap = document.createElement("div");
      wrap.innerHTML = html;
      document.body.insertBefore(wrap.firstElementChild, document.body.firstChild);
    }

    document.querySelectorAll(".navgroup-label").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.preventDefault();
        const group = el.closest(".navgroup");
        if (group) group.classList.toggle("open");
      });
      el.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          const group = el.closest(".navgroup");
          if (group) group.classList.toggle("open");
        }
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", render);
  } else {
    render();
  }
})();
