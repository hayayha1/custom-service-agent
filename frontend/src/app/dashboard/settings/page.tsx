"use client";

import { useEffect, useState } from "react";
import { Activity, CheckCircle2, Cpu, Eye, EyeOff, Layers, Save, Sliders } from "lucide-react";
import DashboardLayout from "@/components/layout/dashboard-layout";
import { api, type ModelSettingsResponse } from "@/lib/api";

export default function SettingsPage() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedMsg, setSavedMsg] = useState<string | null>(null);

  // LLM state
  const [llmBaseUrl, setLlmBaseUrl] = useState("https://api.openai.com/v1");
  const [llmModel, setLlmModel] = useState("gpt-4o-mini");
  const [llmTemperature, setLlmTemperature] = useState("0.2");
  const [memoryModel, setMemoryModel] = useState("");
  const [llmApiKey, setLlmApiKey] = useState("");
  const [llmMasked, setLlmMasked] = useState("");
  const [showLlmKey, setShowLlmKey] = useState(false);

  // Embedding state
  const [embBaseUrl, setEmbBaseUrl] = useState("https://api.openai.com/v1");
  const [embModel, setEmbModel] = useState("text-embedding-3-small");
  const [embDim, setEmbDim] = useState("1536");
  const [embApiKey, setEmbApiKey] = useState("");
  const [embMasked, setEmbMasked] = useState("");
  const [showEmbKey, setShowEmbKey] = useState(false);

  // Reranker state
  const [rrEnabled, setRrEnabled] = useState(false);
  const [rrBaseUrl, setRrBaseUrl] = useState("https://api.siliconflow.cn/v1/rerank");
  const [rrModel, setRrModel] = useState("BAAI/bge-reranker-v2-m3");
  const [rrTopN, setRrTopN] = useState("4");
  const [rrMinScore, setRrMinScore] = useState("0.35");
  const [rrApiKey, setRrApiKey] = useState("");
  const [rrMasked, setRrMasked] = useState("");
  const [showRrKey, setShowRrKey] = useState(false);

  // Langfuse state
  const [lfEnabled, setLfEnabled] = useState(false);
  const [lfHost, setLfHost] = useState("https://cloud.langfuse.com");
  const [lfPublicKey, setLfPublicKey] = useState("");
  const [lfSecretKey, setLfSecretKey] = useState("");
  const [lfPublicMasked, setLfPublicMasked] = useState("");
  const [lfSecretMasked, setLfSecretMasked] = useState("");
  const [showLfKey, setShowLfKey] = useState(false);

  const applyResponse = (data: ModelSettingsResponse) => {
    setLlmBaseUrl(data.llm.base_url);
    setLlmModel(data.llm.model);
    setLlmTemperature(String(data.llm.temperature));
    setMemoryModel(data.llm.memory_model || "");
    setLlmMasked(data.llm.has_api_key ? data.llm.api_key_masked : "");
    setLlmApiKey("");

    setEmbBaseUrl(data.embedding.base_url);
    setEmbModel(data.embedding.model);
    setEmbDim(String(data.embedding.dim));
    setEmbMasked(data.embedding.has_api_key ? data.embedding.api_key_masked : "");
    setEmbApiKey("");

    setRrEnabled(data.reranker.enabled);
    setRrBaseUrl(data.reranker.base_url);
    setRrModel(data.reranker.model);
    setRrTopN(String(data.reranker.top_n));
    setRrMinScore(String(data.reranker.min_score));
    setRrMasked(data.reranker.has_api_key ? data.reranker.api_key_masked : "");
    setRrApiKey("");

    if (data.langfuse) {
      setLfEnabled(data.langfuse.enabled);
      setLfHost(data.langfuse.host || "https://cloud.langfuse.com");
      setLfPublicMasked(data.langfuse.public_key_masked || "");
      setLfSecretMasked(data.langfuse.secret_key_masked || "");
      setLfPublicKey("");
      setLfSecretKey("");
    }
  };

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await api.getModelSettings();
        applyResponse(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "加载模型设置失败");
      } finally {
        setLoading(false);
      }
    };
    void load();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    setSavedMsg(null);
    try {
      const updated = await api.updateModelSettings({
        llm: {
          base_url: llmBaseUrl.trim(),
          model: llmModel.trim(),
          temperature: Number(llmTemperature) || 0.2,
          memory_model: memoryModel.trim(),
          api_key: llmApiKey.trim() || undefined,
        },
        embedding: {
          base_url: embBaseUrl.trim(),
          model: embModel.trim(),
          dim: Number(embDim) || 1536,
          api_key: embApiKey.trim() || undefined,
        },
        reranker: {
          enabled: rrEnabled,
          base_url: rrBaseUrl.trim(),
          model: rrModel.trim(),
          top_n: Number(rrTopN) || 4,
          min_score: Number(rrMinScore) || 0.35,
          api_key: rrApiKey.trim() || undefined,
        },
        langfuse: {
          enabled: lfEnabled,
          host: lfHost.trim() || "https://cloud.langfuse.com",
          public_key: lfPublicKey.trim() || undefined,
          secret_key: lfSecretKey.trim() || undefined,
        },
      });
      applyResponse(updated);
      setSavedMsg("模型配置已保存并热更新生效。");
    } catch (err) {
      setError(err instanceof Error ? err.message : "保存配置失败");
    } finally {
      setSaving(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="max-w-4xl space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">模型设置</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            统一配置垂直客服 Agent 的 LLM 对话模型、Embedding 向量模型与 Reranker 重排模型
          </p>
        </div>

        {error && (
          <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
        )}
        {savedMsg && (
          <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-700 flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4" />
            {savedMsg}
          </div>
        )}

        {loading ? (
          <div className="rounded-xl border bg-card p-10 text-center text-sm text-muted-foreground">
            正在加载当前模型配置...
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* 1. LLM Config Card */}
            <section className="rounded-xl border bg-card p-6 shadow-sm space-y-4">
              <div className="flex items-center justify-between border-b pb-3">
                <div className="flex items-center gap-2.5">
                  <Cpu className="h-5 w-5 text-primary" />
                  <div>
                    <h2 className="text-base font-semibold">LLM 对话模型配置</h2>
                    <p className="text-xs text-muted-foreground">
                      支持所有 OpenAI 兼容接口（OpenAI / DeepSeek / 通义千问 / vLLM）
                    </p>
                  </div>
                </div>
                {llmMasked && (
                  <span className="rounded-full bg-emerald-500/10 px-2.5 py-1 text-xs font-mono text-emerald-700">
                    Key: {llmMasked}
                  </span>
                )}
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-1.5 sm:col-span-2">
                  <label className="text-sm font-medium">Base URL</label>
                  <input
                    type="url"
                    required
                    value={llmBaseUrl}
                    onChange={(e) => setLlmBaseUrl(e.target.value)}
                    placeholder="https://api.openai.com/v1"
                    className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-sm font-medium">主对话模型（Model）</label>
                  <input
                    type="text"
                    required
                    value={llmModel}
                    onChange={(e) => setLlmModel(e.target.value)}
                    placeholder="gpt-4o-mini / deepseek-chat"
                    className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-sm font-medium">记忆提取模型（可选）</label>
                  <input
                    type="text"
                    value={memoryModel}
                    onChange={(e) => setMemoryModel(e.target.value)}
                    placeholder="留空则复用主对话模型"
                    className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Temperature（0 ~ 2）</label>
                  <input
                    type="number"
                    step="0.1"
                    min="0"
                    max="2"
                    value={llmTemperature}
                    onChange={(e) => setLlmTemperature(e.target.value)}
                    className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-sm font-medium">API Key（留空保留原密钥）</label>
                  <div className="relative">
                    <input
                      type={showLlmKey ? "text" : "password"}
                      value={llmApiKey}
                      onChange={(e) => setLlmApiKey(e.target.value)}
                      placeholder={llmMasked ? "已配置，输入新 Key 可覆盖" : "sk-..."}
                      className="w-full rounded-lg border bg-background px-3 py-2 pr-10 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                    />
                    <button
                      type="button"
                      onClick={() => setShowLlmKey(!showLlmKey)}
                      className="absolute right-2.5 top-2.5 text-muted-foreground hover:text-foreground"
                    >
                      {showLlmKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>
              </div>
            </section>

            {/* 2. Embedding Config Card */}
            <section className="rounded-xl border bg-card p-6 shadow-sm space-y-4">
              <div className="flex items-center justify-between border-b pb-3">
                <div className="flex items-center gap-2.5">
                  <Layers className="h-5 w-5 text-primary" />
                  <div>
                    <h2 className="text-base font-semibold">Embedding 向量模型配置</h2>
                    <p className="text-xs text-muted-foreground">
                      用于 pgvector 知识库文档入库与跨会话情景记忆召回
                    </p>
                  </div>
                </div>
                {embMasked && (
                  <span className="rounded-full bg-emerald-500/10 px-2.5 py-1 text-xs font-mono text-emerald-700">
                    Key: {embMasked}
                  </span>
                )}
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-1.5 sm:col-span-2">
                  <label className="text-sm font-medium">Base URL</label>
                  <input
                    type="url"
                    required
                    value={embBaseUrl}
                    onChange={(e) => setEmbBaseUrl(e.target.value)}
                    placeholder="https://api.openai.com/v1"
                    className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Embedding 模型</label>
                  <input
                    type="text"
                    required
                    value={embModel}
                    onChange={(e) => setEmbModel(e.target.value)}
                    placeholder="text-embedding-3-small / BAAI/bge-m3"
                    className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-sm font-medium">向量维度（Dimensions）</label>
                  <input
                    type="number"
                    min="64"
                    max="8192"
                    required
                    value={embDim}
                    onChange={(e) => setEmbDim(e.target.value)}
                    className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>

                <div className="space-y-1.5 sm:col-span-2">
                  <label className="text-sm font-medium">API Key（留空保留原密钥或复用 LLM Key）</label>
                  <div className="relative">
                    <input
                      type={showEmbKey ? "text" : "password"}
                      value={embApiKey}
                      onChange={(e) => setEmbApiKey(e.target.value)}
                      placeholder={embMasked ? "已配置，输入新 Key 可覆盖" : "sk-..."}
                      className="w-full rounded-lg border bg-background px-3 py-2 pr-10 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                    />
                    <button
                      type="button"
                      onClick={() => setShowEmbKey(!showEmbKey)}
                      className="absolute right-2.5 top-2.5 text-muted-foreground hover:text-foreground"
                    >
                      {showEmbKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>
              </div>
            </section>

            {/* 3. Reranker Config Card */}
            <section className="rounded-xl border bg-card p-6 shadow-sm space-y-4">
              <div className="flex items-center justify-between border-b pb-3">
                <div className="flex items-center gap-2.5">
                  <Sliders className="h-5 w-5 text-primary" />
                  <div>
                    <h2 className="text-base font-semibold">Reranker 重排模型配置</h2>
                    <p className="text-xs text-muted-foreground">
                      兼容 SiliconFlow / Jina / Cohere / DashScope 标准 /rerank 接口
                    </p>
                  </div>
                </div>

                <label className="inline-flex items-center gap-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={rrEnabled}
                    onChange={(e) => setRrEnabled(e.target.checked)}
                    className="h-4 w-4 rounded border-input accent-primary"
                  />
                  <span className="text-sm font-medium">
                    {rrEnabled ? "已启用" : "未启用"}
                  </span>
                </label>
              </div>

              <div className={`grid gap-4 sm:grid-cols-2 ${rrEnabled ? "" : "opacity-60"}`}>
                <div className="space-y-1.5 sm:col-span-2">
                  <label className="text-sm font-medium">Rerank Endpoint URL</label>
                  <input
                    type="url"
                    value={rrBaseUrl}
                    onChange={(e) => setRrBaseUrl(e.target.value)}
                    placeholder="https://api.siliconflow.cn/v1/rerank"
                    className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Reranker 模型</label>
                  <input
                    type="text"
                    value={rrModel}
                    onChange={(e) => setRrModel(e.target.value)}
                    placeholder="BAAI/bge-reranker-v2-m3"
                    className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-sm font-medium">API Key（留空保留原密钥）</label>
                  <div className="relative">
                    <input
                      type={showRrKey ? "text" : "password"}
                      value={rrApiKey}
                      onChange={(e) => setRrApiKey(e.target.value)}
                      placeholder={rrMasked ? `已配置 (${rrMasked})` : "sk-..."}
                      className="w-full rounded-lg border bg-background px-3 py-2 pr-10 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                    />
                    <button
                      type="button"
                      onClick={() => setShowRrKey(!showRrKey)}
                      className="absolute right-2.5 top-2.5 text-muted-foreground hover:text-foreground"
                    >
                      {showRrKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>

                <div className="space-y-1.5">
                  <label className="text-sm font-medium">重排保留条数（Top N）</label>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={rrTopN}
                    onChange={(e) => setRrTopN(e.target.value)}
                    className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-sm font-medium">最低相关性阈值（0 ~ 1）</label>
                  <input
                    type="number"
                    step="0.05"
                    min="0"
                    max="1"
                    value={rrMinScore}
                    onChange={(e) => setRrMinScore(e.target.value)}
                    className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>
              </div>
            </section>

            {/* 4. Langfuse Observability Config Card */}
            <section className="rounded-xl border bg-card p-6 shadow-sm space-y-4">
              <div className="flex items-center justify-between border-b pb-3">
                <div className="flex items-center gap-2.5">
                  <Activity className="h-5 w-5 text-primary" />
                  <div>
                    <h2 className="text-base font-semibold">Langfuse 全链路监测配置</h2>
                    <p className="text-xs text-muted-foreground">
                      可选同步上报全链路 Trace/Span、工具调用、Token 消耗与 TPS 到 Langfuse Cloud 或私有化实例
                    </p>
                  </div>
                </div>

                <label className="inline-flex items-center gap-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={lfEnabled}
                    onChange={(e) => setLfEnabled(e.target.checked)}
                    className="h-4 w-4 rounded border-input accent-primary"
                  />
                  <span className="text-sm font-medium">
                    {lfEnabled ? "已启用" : "未启用"}
                  </span>
                </label>
              </div>

              <div className={`grid gap-4 sm:grid-cols-2 ${lfEnabled ? "" : "opacity-60"}`}>
                <div className="space-y-1.5 sm:col-span-2">
                  <label className="text-sm font-medium">Langfuse Host</label>
                  <input
                    type="url"
                    value={lfHost}
                    onChange={(e) => setLfHost(e.target.value)}
                    placeholder="https://cloud.langfuse.com"
                    className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Public Key（留空保留原公钥）</label>
                  <input
                    type="text"
                    value={lfPublicKey}
                    onChange={(e) => setLfPublicKey(e.target.value)}
                    placeholder={lfPublicMasked ? `已配置 (${lfPublicMasked})` : "pk-lf-..."}
                    className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Secret Key（留空保留原密钥）</label>
                  <div className="relative">
                    <input
                      type={showLfKey ? "text" : "password"}
                      value={lfSecretKey}
                      onChange={(e) => setLfSecretKey(e.target.value)}
                      placeholder={lfSecretMasked ? `已配置 (${lfSecretMasked})` : "sk-lf-..."}
                      className="w-full rounded-lg border bg-background px-3 py-2 pr-10 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                    />
                    <button
                      type="button"
                      onClick={() => setShowLfKey(!showLfKey)}
                      className="absolute right-2.5 top-2.5 text-muted-foreground hover:text-foreground"
                    >
                      {showLfKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>
              </div>
            </section>

            {/* Submit Bar */}
            <div className="flex justify-end">
              <button
                type="submit"
                disabled={saving}
                className="inline-flex items-center rounded-lg bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground shadow hover:bg-primary/90 transition-colors disabled:opacity-50"
              >
                <Save className="mr-2 h-4 w-4" />
                {saving ? "保存中..." : "保存配置"}
              </button>
            </div>
          </form>
        )}
      </div>
    </DashboardLayout>
  );
}
