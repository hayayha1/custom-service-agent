"use client";

import { useCallback, useEffect, useState } from "react";
import {
  BookOpen,
  CheckCircle2,
  Layers,
  Plus,
  Search,
  Trash2,
  X,
} from "lucide-react";
import DashboardLayout from "@/components/layout/dashboard-layout";
import { api, type KBDocumentItem, type KBSearchResponse } from "@/lib/api";

const CATEGORIES = [
  { value: "", label: "全部分类" },
  { value: "return_policy", label: "退换货政策" },
  { value: "warranty", label: "质保维修" },
  { value: "shipping", label: "物流发货" },
  { value: "invoice", label: "发票规则" },
  { value: "general", label: "通用规则" },
];

export default function KnowledgeBasePage() {
  const [documents, setDocuments] = useState<KBDocumentItem[]>([]);
  const [selectedCategory, setSelectedCategory] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [ingestFeedback, setIngestFeedback] = useState<string | null>(null);

  // Add/Edit modal state
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [docKey, setDocKey] = useState("");
  const [category, setCategory] = useState("return_policy");
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [chunkMode, setChunkMode] = useState<"auto" | "sop" | "faq">("auto");
  const [chunkSize, setChunkSize] = useState("450");
  const [redactPii, setRedactPii] = useState(true);
  const [saving, setSaving] = useState(false);

  // 4-stage Hybrid Search test state
  const [searchQuery, setSearchQuery] = useState("");
  const [slotProduct, setSlotProduct] = useState("");
  const [searching, setSearching] = useState(false);
  const [searchResult, setSearchResult] = useState<KBSearchResponse | null>(null);

  const loadDocuments = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.listKBDocuments(selectedCategory || undefined);
      setDocuments(res.documents);
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载知识库失败");
    } finally {
      setLoading(false);
    }
  }, [selectedCategory]);

  useEffect(() => {
    void loadDocuments();
  }, [loadDocuments]);

  const openCreateModal = () => {
    setDocKey("");
    setCategory(selectedCategory || "return_policy");
    setTitle("");
    setContent("");
    setChunkMode("auto");
    setChunkSize("450");
    setRedactPii(true);
    setIsModalOpen(true);
  };

  const openEditModal = (doc: KBDocumentItem) => {
    setDocKey(doc.doc_key);
    setCategory(doc.category);
    setTitle(doc.title);
    setContent(doc.content);
    setChunkMode("auto");
    setIsModalOpen(true);
  };

  const handleSaveDoc = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim() || !content.trim()) return;
    setSaving(true);
    setError(null);
    setIngestFeedback(null);
    try {
      const res = await api.upsertKBDocument({
        doc_key: docKey.trim() || undefined,
        category,
        title: title.trim(),
        content: content.trim(),
        chunk_mode: chunkMode,
        chunk_size: Number(chunkSize) || 450,
        redact_pii: redactPii,
      });
      setIsModalOpen(false);
      setIngestFeedback(
        res.skipped_unchanged
          ? `文档 "${res.title}" 内容指纹 (${res.content_hash.slice(0, 8)}) 未变化，已跳过重复向量化（共 ${res.chunk_count} 个 Chunk）`
          : `文档 "${res.title}" 入库完成：切分 ${res.chunk_count} 个 Chunk，清洗噪声 ${res.noise_removed_count} 处，脱敏敏感信息 ${res.pii_redacted_count} 处`
      );
      await loadDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "保存文档失败");
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteDoc = async (key: string) => {
    if (!window.confirm(`确认删除知识文档 "${key}" 及其所有检索分块？`)) return;
    try {
      await api.deleteKBDocument(key);
      await loadDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "删除失败");
    }
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    setSearching(true);
    setError(null);
    try {
      const slots = slotProduct.trim() ? { product_name: slotProduct.trim() } : undefined;
      const res = await api.searchKB({
        query: searchQuery.trim(),
        category: selectedCategory || null,
        top_k: 4,
        slots,
      });
      setSearchResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "混合检索失败");
    } finally {
      setSearching(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">知识库构建与混合检索</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              数据清洗脱敏、结构化 Chunk 分块（SOP 标题面包屑 / FAQ 问答对），以及四阶段混合检索（Query改写 → 向量+全文 → RRF → Rerank）
            </p>
          </div>
          <button
            onClick={openCreateModal}
            className="inline-flex items-center justify-center rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground shadow hover:bg-primary/90 transition-colors"
          >
            <Plus className="mr-2 h-4 w-4" />
            清洗并导入文档
          </button>
        </div>

        {error && (
          <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
        )}
        {ingestFeedback && (
          <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-700 flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 shrink-0" />
            <span>{ingestFeedback}</span>
          </div>
        )}

        {/* 4-Stage Hybrid Retrieval Test Card */}
        <div className="rounded-xl border bg-card p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm font-semibold">
              <Search className="h-4 w-4 text-primary" />
              四阶段混合检索测试（Query 改写 → Dense + Sparse → RRF 融合 → Rerank 重排）
            </div>
          </div>
          <form onSubmit={handleSearch} className="grid gap-3 sm:grid-cols-12">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="输入用户提问（支持代词与口语），如：它单耳没声音坏了能换新吗？邮费谁掏？"
              className="sm:col-span-7 rounded-lg border bg-background px-3.5 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            />
            <input
              type="text"
              value={slotProduct}
              onChange={(e) => setSlotProduct(e.target.value)}
              placeholder="模拟会话槽位商品（如：声学降噪耳机 Pro）"
              className="sm:col-span-3 rounded-lg border bg-background px-3 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-ring"
            />
            <button
              type="submit"
              disabled={searching || !searchQuery.trim()}
              className="sm:col-span-2 inline-flex items-center justify-center rounded-lg bg-secondary px-4 py-2 text-sm font-medium text-secondary-foreground hover:bg-secondary/80 transition-colors disabled:opacity-50"
            >
              {searching ? "检索中..." : "执行混合检索"}
            </button>
          </form>

          {searchResult !== null && (
            <div className="space-y-3 pt-3 border-t">
              {/* Pipeline Diagnostics Bar */}
              <div className="rounded-lg bg-muted/50 p-3 text-xs space-y-1.5">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">Stage 1 Query 改写：</span>
                  <code className="rounded bg-background px-2 py-0.5 font-mono">
                    {searchResult.rewritten_query || searchResult.query}
                  </code>
                  {searchResult.rewrite_applied && (
                    <span className="rounded bg-primary/10 text-primary px-1.5 py-0.5 text-[10px]">
                      已结合槽位/同义词改写
                    </span>
                  )}
                </div>
                <div className="flex flex-wrap items-center gap-3 text-muted-foreground">
                  <span>Stage 2A 向量召回: {searchResult.dense_count ?? 0} 条</span>
                  <span>·</span>
                  <span>Stage 2B 全文(GIN)召回: {searchResult.sparse_count ?? 0} 条</span>
                  <span>·</span>
                  <span>Stage 3 RRF 融合: {searchResult.fused_count ?? 0} 条</span>
                  <span>·</span>
                  <span>
                    Stage 4 Rerank 精排: {searchResult.reranked ? "已启用" : "未启用(使用RRF排序)"}
                  </span>
                  {searchResult.timings && (
                    <>
                      <span>·</span>
                      <span className="font-mono">
                        总耗时 {searchResult.timings.total_ms}ms (改写{" "}
                        {searchResult.timings.rewrite_ms}ms / 向量 {searchResult.timings.dense_ms}ms
                        / 全文 {searchResult.timings.sparse_ms}ms / RRF{" "}
                        {searchResult.timings.rrf_ms}ms / Rerank {searchResult.timings.rerank_ms}ms)
                      </span>
                    </>
                  )}
                </div>
              </div>

              {searchResult.matches.length === 0 ? (
                <div className="text-sm text-muted-foreground py-2">
                  未命中高置信度知识分块（Zero-Hit 拦截生效）
                </div>
              ) : (
                <div className="grid gap-3 md:grid-cols-2">
                  {searchResult.matches.map((m, idx) => (
                    <div
                      key={m.chunk_key || `${m.doc_key}-${idx}`}
                      className="rounded-lg border bg-muted/20 p-3.5 space-y-2"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <div className="font-medium text-sm">{m.title}</div>
                          {m.section_path && (
                            <div className="text-[11px] text-primary font-medium">
                              {m.section_path}
                            </div>
                          )}
                        </div>
                        <span className="shrink-0 rounded bg-primary/10 px-2 py-0.5 font-mono text-xs text-primary">
                          score: {m.score}
                        </span>
                      </div>
                      <p className="text-xs text-muted-foreground whitespace-pre-wrap line-clamp-4">
                        {m.content}
                      </p>
                      <div className="flex flex-wrap gap-2 text-[10px] font-mono text-muted-foreground pt-1 border-t">
                        <span>Dense: {m.dense_score ?? "--"}</span>
                        <span>Sparse: {m.sparse_score ?? "--"}</span>
                        <span>RRF: {m.rrf_score ?? "--"}</span>
                        {m.rerank_score !== null && m.rerank_score !== undefined && (
                          <span className="text-emerald-700 font-semibold">
                            Rerank: {m.rerank_score}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Category Filter Tabs */}
        <div className="flex flex-wrap gap-2">
          {CATEGORIES.map((cat) => (
            <button
              key={cat.value || "all"}
              onClick={() => setSelectedCategory(cat.value)}
              className={`rounded-lg px-3.5 py-1.5 text-sm font-medium transition-colors ${
                selectedCategory === cat.value
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "bg-muted text-muted-foreground hover:text-foreground"
              }`}
            >
              {cat.label}
            </button>
          ))}
        </div>

        {/* Document List */}
        {loading ? (
          <div className="rounded-xl border bg-card p-10 text-center text-sm text-muted-foreground">
            正在加载知识库文档...
          </div>
        ) : documents.length === 0 ? (
          <div className="rounded-xl border bg-card p-12 text-center space-y-3">
            <BookOpen className="mx-auto h-10 w-10 text-muted-foreground/60" />
            <div className="text-base font-medium">暂无知识库文档</div>
            <p className="text-sm text-muted-foreground">
              点击右上角“清洗并导入文档”，或运行{" "}
              <code className="rounded bg-muted px-1.5 py-0.5">
                uv run python -m scripts.seed_kb
              </code>{" "}
              导入示例政策。
            </p>
          </div>
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            {documents.map((doc) => (
              <div
                key={doc.doc_key}
                className="rounded-xl border bg-card p-5 shadow-sm flex flex-col justify-between gap-4 hover:shadow-md transition-shadow"
              >
                <div className="space-y-2">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h3 className="font-semibold text-base">{doc.title}</h3>
                      <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                        <span className="rounded bg-secondary px-2 py-0.5 text-secondary-foreground">
                          {doc.category}
                        </span>
                        <span className="font-mono">{doc.doc_key}</span>
                        {doc.metadata?.chunk_count && (
                          <span className="inline-flex items-center rounded bg-primary/10 text-primary px-2 py-0.5">
                            <Layers className="mr-1 h-3 w-3" />
                            {doc.metadata.chunk_count} Chunks
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => openEditModal(doc)}
                        className="rounded-md px-2.5 py-1 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-foreground"
                      >
                        编辑
                      </button>
                      <button
                        onClick={() => void handleDeleteDoc(doc.doc_key)}
                        className="rounded-md p-1.5 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                        title="删除文档"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground whitespace-pre-wrap leading-relaxed">
                    {doc.content}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Add / Edit Ingestion Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-2xl rounded-xl border bg-card p-6 shadow-lg space-y-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">
                {docKey ? "编辑并重新分块入库" : "清洗并导入知识文档"}
              </h2>
              <button
                onClick={() => setIsModalOpen(false)}
                className="rounded-md p-1 hover:bg-accent"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <form onSubmit={handleSaveDoc} className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">文档标识（可选）</label>
                  <input
                    type="text"
                    value={docKey}
                    onChange={(e) => setDocKey(e.target.value)}
                    placeholder="留空自动生成，如 policy_return_7day"
                    className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">业务分类</label>
                  <select
                    value={category}
                    onChange={(e) => setCategory(e.target.value)}
                    className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  >
                    {CATEGORIES.filter((c) => c.value).map((c) => (
                      <option key={c.value} value={c.value}>
                        {c.label} ({c.value})
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Chunk 切分策略</label>
                  <select
                    value={chunkMode}
                    onChange={(e) => setChunkMode(e.target.value as "auto" | "sop" | "faq")}
                    className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  >
                    <option value="auto">自动识别 (Auto)</option>
                    <option value="sop">SOP 标题层级 + 滑窗</option>
                    <option value="faq">FAQ 问答对整块保留</option>
                  </select>
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">最大 Chunk 字数</label>
                  <input
                    type="number"
                    min="100"
                    max="2000"
                    value={chunkSize}
                    onChange={(e) => setChunkSize(e.target.value)}
                    className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>
                <div className="flex items-end pb-2">
                  <label className="inline-flex items-center gap-2 cursor-pointer text-sm">
                    <input
                      type="checkbox"
                      checked={redactPii}
                      onChange={(e) => setRedactPii(e.target.checked)}
                      className="h-4 w-4 rounded border-input accent-primary"
                    />
                    <span>自动脱敏手机号/身份证/邮箱</span>
                  </label>
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium">标题</label>
                <input
                  type="text"
                  required
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="例如：7天无理由退换货政策"
                  className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium">
                  正文内容（支持 Markdown 标题 `# / ##` 或 `Q1: / A1:` 问答对，自动剥离 HTML 噪声）
                </label>
                <textarea
                  required
                  rows={8}
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  placeholder="输入详细的售后政策、FAQ 或 SOP 规则..."
                  className="w-full rounded-lg border bg-background px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-ring"
                />
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setIsModalOpen(false)}
                  className="rounded-lg border px-4 py-2 text-sm font-medium hover:bg-accent"
                >
                  取消
                </button>
                <button
                  type="submit"
                  disabled={saving}
                  className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                >
                  {saving ? "清洗分块与向量化中..." : "执行清洗、分块并入库"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </DashboardLayout>
  );
}
