from pathlib import Path

root = Path(__file__).resolve().parents[1]
legacy_path = root.parent / "index.html"
target_path = root / "templates" / "index.html"

html = legacy_path.read_text(encoding="utf-8")

# 标题更新
html = html.replace("<title>批量TXT转XML格式化工具</title>", "<title>AI Tag 工具箱</title>", 1)
html = html.replace("<h1>批量TXT转XML格式化工具</h1>", "<h1>AI Tag 工具箱<span>Prompt XML Studio</span></h1>", 1)

# 添加主题切换按钮
html = html.replace(
    '<button class="config-toggle" onclick="toggleSidebar()">配置面板</button>',
    '<button class="theme-toggle" type="button" onclick="toggleThemeMode()" title="切换黑色/白色主题">◐ 主题</button>\\n            <button class="config-toggle" onclick="toggleSidebar()">配置面板</button>',
    1,
)

discord_glass_css = r"""
/* =========================================================
   Discord-like glass redesign overlay
   - Keeps legacy IDs / JS / backend contracts intact
   - Dark / light theme with purple accent
   ========================================================= */
:root {
    color-scheme: dark;
    --app-bg: #08080d;
    --app-bg-soft: #10101a;
    --app-bg-elevated: rgba(22, 22, 34, 0.70);
    --app-bg-elevated-2: rgba(31, 31, 48, 0.58);
    --app-surface: rgba(28, 28, 44, 0.62);
    --app-surface-solid: #181823;
    --app-surface-hover: rgba(139, 92, 246, 0.12);
    --app-text: #f4f3ff;
    --app-text-muted: #aaa6c3;
    --app-text-soft: #d8d5ee;
    --app-border: rgba(180, 165, 255, 0.16);
    --app-border-strong: rgba(180, 165, 255, 0.28);
    --app-accent: #8b5cf6;
    --app-accent-2: #a78bfa;
    --app-accent-3: #c084fc;
    --app-success: #22c55e;
    --app-warning: #f59e0b;
    --app-danger: #ef4444;
    --app-info: #38bdf8;
    --app-shadow: 0 18px 60px rgba(0, 0, 0, 0.42);
    --app-shadow-soft: 0 10px 32px rgba(0, 0, 0, 0.24);
    --app-blur: blur(22px) saturate(1.25);
    --nav-width: 252px;
    --config-width: 360px;
}

:root[data-theme="light"] {
    color-scheme: light;
    --app-bg: #f4f2ff;
    --app-bg-soft: #ffffff;
    --app-bg-elevated: rgba(255, 255, 255, 0.72);
    --app-bg-elevated-2: rgba(255, 255, 255, 0.58);
    --app-surface: rgba(255, 255, 255, 0.68);
    --app-surface-solid: #ffffff;
    --app-surface-hover: rgba(124, 58, 237, 0.10);
    --app-text: #171426;
    --app-text-muted: #68617f;
    --app-text-soft: #312b48;
    --app-border: rgba(124, 58, 237, 0.16);
    --app-border-strong: rgba(124, 58, 237, 0.28);
    --app-accent: #7c3aed;
    --app-accent-2: #8b5cf6;
    --app-accent-3: #a855f7;
    --app-shadow: 0 18px 60px rgba(79, 70, 229, 0.15);
    --app-shadow-soft: 0 10px 32px rgba(79, 70, 229, 0.10);
}

/* Global */
html,
body {
    height: 100%;
    overflow: hidden;
}

body {
    color: var(--app-text) !important;
    background:
        radial-gradient(circle at 12% 12%, rgba(139, 92, 246, 0.35), transparent 32%),
        radial-gradient(circle at 82% 8%, rgba(192, 132, 252, 0.22), transparent 30%),
        radial-gradient(circle at 78% 84%, rgba(56, 189, 248, 0.12), transparent 34%),
        linear-gradient(135deg, var(--app-bg), var(--app-bg-soft)) !important;
}

body.custom-bg::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: -1;
    background:
        radial-gradient(circle at 20% 20%, rgba(139, 92, 246, 0.30), transparent 32%),
        rgba(7, 7, 12, 0.58);
    pointer-events: none;
}

.container {
    min-height: 100vh !important;
}

/* Left Discord-like rail */
.top-bar {
    position: fixed !important;
    inset: 0 auto 0 0 !important;
    width: var(--nav-width) !important;
    height: 100vh !important;
    padding: 18px 14px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    gap: 14px !important;
    border-right: 1px solid var(--app-border) !important;
    border-bottom: none !important;
    background: rgba(12, 12, 20, 0.58) !important;
    backdrop-filter: var(--app-blur) !important;
    -webkit-backdrop-filter: var(--app-blur) !important;
    box-shadow: var(--app-shadow-soft) !important;
    z-index: 1000 !important;
}

:root[data-theme="light"] .top-bar {
    background: rgba(255, 255, 255, 0.58) !important;
}

.top-bar h1 {
    margin: 0 !important;
    padding: 12px 12px 14px !important;
    color: var(--app-text) !important;
    font-size: 17px !important;
    line-height: 1.18 !important;
    letter-spacing: 0.01em !important;
    border: 1px solid var(--app-border) !important;
    border-radius: 18px !important;
    background:
        linear-gradient(135deg, rgba(139, 92, 246, 0.18), rgba(192, 132, 252, 0.06)),
        var(--app-bg-elevated-2) !important;
    box-shadow: var(--app-shadow-soft) !important;
}

.top-bar h1::before {
    content: "";
    display: inline-block;
    width: 11px;
    height: 11px;
    margin-right: 8px;
    border-radius: 999px;
    background: linear-gradient(135deg, var(--app-accent), var(--app-accent-3));
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.75);
    vertical-align: 1px;
}

.top-bar h1 span {
    display: block;
    margin-top: 4px;
    padding-left: 22px;
    color: var(--app-text-muted);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.top-bar .tabs {
    height: auto !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 5px !important;
    overflow-y: auto !important;
    padding: 4px 2px !important;
    scrollbar-width: thin;
}

.tab {
    height: auto !important;
    line-height: 1.25 !important;
    padding: 11px 12px !important;
    display: flex !important;
    align-items: center !important;
    color: var(--app-text-muted) !important;
    border: 1px solid transparent !important;
    border-radius: 12px !important;
    background: transparent !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    transition: transform 0.16s ease, background 0.16s ease, color 0.16s ease, border-color 0.16s ease !important;
}

.tab:hover {
    color: var(--app-text) !important;
    background: var(--app-surface-hover) !important;
    border-color: var(--app-border) !important;
    transform: translateX(2px);
}

.tab.active {
    color: #fff !important;
    border-color: rgba(167, 139, 250, 0.40) !important;
    background:
        linear-gradient(135deg, rgba(139, 92, 246, 0.92), rgba(124, 58, 237, 0.58)) !important;
    box-shadow: 0 12px 30px rgba(139, 92, 246, 0.28) !important;
}

:root[data-theme="light"] .tab.active {
    color: #fff !important;
}

/* Top rail buttons */
.top-bar .config-toggle,
.top-bar .theme-toggle {
    margin: 0 !important;
    width: 100% !important;
    padding: 10px 12px !important;
    border-radius: 12px !important;
    color: var(--app-text-soft) !important;
    border: 1px solid var(--app-border) !important;
    background: var(--app-bg-elevated-2) !important;
    font-weight: 800 !important;
    letter-spacing: 0.02em !important;
    box-shadow: var(--app-shadow-soft) !important;
}

.top-bar .config-toggle:hover,
.top-bar .theme-toggle:hover {
    color: #fff !important;
    border-color: var(--app-border-strong) !important;
    background: linear-gradient(135deg, var(--app-accent), var(--app-accent-2)) !important;
}

/* Main layout */
.main-layout {
    height: 100vh !important;
    margin-left: var(--nav-width) !important;
    display: flex !important;
    background: transparent !important;
}

.sidebar {
    width: var(--config-width) !important;
    min-width: var(--config-width) !important;
    padding: 16px !important;
    color: var(--app-text) !important;
    border-right: 1px solid var(--app-border) !important;
    background: rgba(14, 14, 24, 0.42) !important;
    backdrop-filter: var(--app-blur) !important;
    -webkit-backdrop-filter: var(--app-blur) !important;
    box-shadow: var(--app-shadow-soft) !important;
}

:root[data-theme="light"] .sidebar {
    background: rgba(255, 255, 255, 0.46) !important;
}

.sidebar.collapsed {
    margin-left: calc(-1 * var(--config-width)) !important;
}

.sidebar h2,
.config-section h2,
.result-section h3,
.tab-content h2,
.tab-content h3 {
    color: var(--app-text) !important;
}

.main-panel {
    padding: 18px !important;
    background: transparent !important;
}

.tab-content.active {
    animation: tabFadeIn 0.18s ease !important;
}

/* Cards / sections */
.config-section,
.result,
.preview-card,
#mask_pipeline_status,
#cl_model_status,
#preview_stats,
#training_cleanup,
#training_summary > div > div,
#mm_sam2_panel,
#mm_canvas_container + div,
#manualmask-tab > div > div:first-child,
#preview_modal > div {
    color: var(--app-text) !important;
    border: 1px solid var(--app-border) !important;
    background: var(--app-surface) !important;
    backdrop-filter: var(--app-blur) !important;
    -webkit-backdrop-filter: var(--app-blur) !important;
    box-shadow: var(--app-shadow-soft) !important;
}

.config-section {
    border-radius: 18px !important;
    padding: 16px !important;
}

.two-col {
    gap: 18px !important;
}

.result {
    min-height: 180px;
    max-height: calc(100vh - 132px) !important;
    border-radius: 18px !important;
    color: var(--app-text-soft) !important;
    font-family: "Cascadia Code", "JetBrains Mono", "SF Mono", Consolas, monospace !important;
    background: rgba(8, 8, 14, 0.56) !important;
}

:root[data-theme="light"] .result {
    background: rgba(255, 255, 255, 0.64) !important;
}

.result:empty::before {
    color: var(--app-text-muted) !important;
}

/* Forms */
label,
.form-group label,
p,
span,
div {
    color: inherit;
}

label,
.form-group label,
.config-section p,
#mask_pipeline_status,
#cl_model_status {
    color: var(--app-text-muted) !important;
}

input[type="text"],
input[type="number"],
input[type="password"],
textarea,
select,
input[type="file"] {
    color: var(--app-text) !important;
    border: 1px solid var(--app-border) !important;
    border-radius: 12px !important;
    background: rgba(10, 10, 18, 0.56) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.035) !important;
}

:root[data-theme="light"] input[type="text"],
:root[data-theme="light"] input[type="number"],
:root[data-theme="light"] input[type="password"],
:root[data-theme="light"] textarea,
:root[data-theme="light"] select,
:root[data-theme="light"] input[type="file"] {
    background: rgba(255, 255, 255, 0.76) !important;
}

input[type="text"]:focus,
input[type="number"]:focus,
input[type="password"]:focus,
textarea:focus,
select:focus {
    border-color: rgba(167, 139, 250, 0.74) !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.18) !important;
}

input[type="checkbox"],
input[type="radio"] {
    accent-color: var(--app-accent);
}

input[type="range"] {
    accent-color: var(--app-accent);
}

/* Buttons */
button,
.file-input-label {
    border-radius: 12px !important;
    font-weight: 800 !important;
    letter-spacing: 0.01em !important;
}

.btn-primary,
button.btn-primary,
.file-input-label {
    color: #fff !important;
    background: linear-gradient(135deg, var(--app-accent), var(--app-accent-2)) !important;
    box-shadow: 0 10px 26px rgba(139, 92, 246, 0.28) !important;
}

.btn-primary:hover,
button.btn-primary:hover,
.file-input-label:hover {
    transform: translateY(-1px);
    background: linear-gradient(135deg, var(--app-accent-2), var(--app-accent-3)) !important;
}

.btn-secondary,
button.btn-secondary {
    color: #fff !important;
    background: linear-gradient(135deg, #3b82f6, var(--app-accent)) !important;
}

.btn-tertiary,
button.btn-tertiary {
    color: #fff !important;
    background: linear-gradient(135deg, #f97316, #ef4444) !important;
}

.btn-reset-bg,
button[style*="background:#ef4444"],
button[style*="background:#e53e3e"],
button[style*="background:#dc2626"] {
    color: #fff !important;
    background: linear-gradient(135deg, #ef4444, #be123c) !important;
}

/* Preview */
#preview_grid {
    gap: 14px !important;
}

.preview-card {
    border-radius: 18px !important;
    overflow: hidden !important;
    background: var(--app-surface) !important;
}

.preview-card:hover {
    border-color: var(--app-border-strong) !important;
    box-shadow: 0 18px 42px rgba(139, 92, 246, 0.18) !important;
    transform: translateY(-2px);
}

.preview-card .card-img {
    background: rgba(0, 0, 0, 0.22) !important;
}

.preview-card .card-title {
    color: var(--app-text) !important;
}

.preview-card .card-tags {
    color: var(--app-text-muted) !important;
}

.pagination-bar button {
    color: var(--app-text-soft) !important;
    border-color: var(--app-border) !important;
    background: var(--app-surface) !important;
}

.pagination-bar button.active {
    color: #fff !important;
    border-color: transparent !important;
    background: linear-gradient(135deg, var(--app-accent), var(--app-accent-2)) !important;
}

/* Modals */
#preview_modal {
    background: rgba(0, 0, 0, 0.66) !important;
    backdrop-filter: blur(14px) !important;
}

#preview_modal > div {
    border-radius: 22px !important;
}

#preview_modal [style*="background:#fff"],
#preview_modal [style*="background:#fafafa"] {
    background: var(--app-surface) !important;
}

#modal_filename,
#modal_tags {
    color: var(--app-text) !important;
}

/* Manual mask editor */
#manualmask-tab > div {
    height: calc(100vh - 36px) !important;
    border: 1px solid var(--app-border) !important;
    border-radius: 22px !important;
    overflow: hidden !important;
    background: var(--app-surface) !important;
    box-shadow: var(--app-shadow-soft) !important;
}

#manualmask-tab > div > div:first-child {
    background: rgba(18, 18, 30, 0.74) !important;
}

:root[data-theme="light"] #manualmask-tab > div > div:first-child {
    background: rgba(255, 255, 255, 0.78) !important;
}

#mm_canvas_container {
    background:
        linear-gradient(45deg, rgba(255,255,255,0.03) 25%, transparent 25%),
        linear-gradient(-45deg, rgba(255,255,255,0.03) 25%, transparent 25%),
        #0b0b12 !important;
    background-size: 18px 18px !important;
}

#mm_canvas_container + div,
#manualmask-tab [style*="background:#f7f8fa"],
#manualmask-tab [style*="background:#fff"] {
    background: var(--app-surface) !important;
    color: var(--app-text) !important;
    border-color: var(--app-border) !important;
}

/* Inline legacy color normalization */
[style*="color:#111"],
[style*="color:#1a1a1a"],
[style*="color:#333"],
[style*="color:#444"],
[style*="color:#555"],
[style*="color:#666"],
[style*="color:#888"],
[style*="color:#999"],
[style*="color:#bbb"] {
    color: var(--app-text-muted) !important;
}

[style*="background:#f7f8fa"],
[style*="background:#fff"],
[style*="background:#fafafa"],
[style*="background:#f0f2f5"] {
    background: var(--app-surface) !important;
}

[style*="border:1px solid #e2e5ea"],
[style*="border:1px solid #e5e5e5"],
[style*="border-bottom:1px solid #e5e5e5"],
[style*="border-right:1px solid #e5e5e5"],
[style*="border-left:1px solid #e2e5ea"] {
    border-color: var(--app-border) !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 9px;
    height: 9px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    border: 2px solid transparent;
    border-radius: 999px;
    background: rgba(139, 92, 246, 0.42);
    background-clip: padding-box;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(167, 139, 250, 0.68);
    background-clip: padding-box;
}

/* Responsive */
@media (max-width: 1120px) {
    :root {
        --nav-width: 78px;
        --config-width: 320px;
    }

    .top-bar h1 {
        font-size: 0 !important;
        padding: 14px 10px !important;
        text-align: center;
    }

    .top-bar h1::before {
        width: 24px;
        height: 24px;
        margin: 0;
    }

    .top-bar h1 span {
        display: none;
    }

    .tab {
        justify-content: center !important;
        padding: 11px 6px !important;
        font-size: 0 !important;
    }

    .tab::first-letter {
        font-size: 16px;
    }

    .top-bar .config-toggle,
    .top-bar .theme-toggle {
        font-size: 0 !important;
        padding: 11px 6px !important;
    }

    .top-bar .theme-toggle::before {
        content: "◐";
        font-size: 15px;
    }

    .top-bar .config-toggle::before {
        content: "⚙";
        font-size: 15px;
    }
}

@media (max-width: 900px) {
    html,
    body {
        overflow: auto;
    }

    .top-bar {
        position: sticky !important;
        width: 100% !important;
        height: auto !important;
        inset: 0 auto auto 0 !important;
        flex-direction: row !important;
        overflow-x: auto !important;
    }

    .top-bar h1 {
        min-width: 180px;
        font-size: 15px !important;
    }

    .top-bar h1 span {
        display: block;
    }

    .top-bar .tabs {
        flex-direction: row !important;
        min-width: max-content;
    }

    .tab {
        font-size: 12px !important;
        min-width: max-content;
    }

    .main-layout {
        margin-left: 0 !important;
        height: auto !important;
        min-height: calc(100vh - 76px);
    }

    .sidebar {
        position: fixed !important;
        inset: 76px auto 0 0;
        z-index: 999;
        max-width: 86vw;
    }

    .main-panel {
        min-height: calc(100vh - 76px);
    }
}
"""

# 将重设计 CSS 作为覆盖层插入旧 style 末尾
html = html.replace("</style>", discord_glass_css + "\n    </style>", 1)

theme_script = r"""
        // ===== Discord-like theme toggle =====
        (function initThemeMode() {
            const saved = localStorage.getItem('prompt_xml_theme') || 'dark';
            document.documentElement.setAttribute('data-theme', saved);
        })();

        function toggleThemeMode() {
            const root = document.documentElement;
            const current = root.getAttribute('data-theme') || 'dark';
            const next = current === 'dark' ? 'light' : 'dark';
            root.setAttribute('data-theme', next);
            localStorage.setItem('prompt_xml_theme', next);
        }

"""
html = html.replace("        // sidebar toggle", theme_script + "        // sidebar toggle", 1)

# 增强加载背景函数：保留旧功能，无需替换 JS 逻辑
target_path.write_text(html, encoding="utf-8")
print(f"Rebuilt {target_path} from {legacy_path}")
print(f"Output size: {target_path.stat().st_size} bytes")
