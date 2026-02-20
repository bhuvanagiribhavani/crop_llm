/**
 * ============================================================================
 * LLM INSIGHTS COMPONENT
 * ============================================================================
 * Fetches the latest prediction, calls POST /llm-insight/{id}, and displays:
 *   - Crop Type, NDVI, Yield (tons)
 *   - AI-generated insight paragraph
 *   - Recommendation bullet points
 *   - "Generated using Mistral-7B via Ollama" badge
 */

import React, { useState, useCallback } from 'react';
import axios from 'axios';
import {
  Sparkles,
  Leaf,
  TrendingUp,
  Wheat,
  AlertCircle,
  Loader2,
  CheckCircle,
  RefreshCw,
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:5000';

const LLMInsights = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [data, setData] = useState(null); // { crop_type, ndvi, estimated_yield_tons, insight, recommendations }

  /**
   * Fetch the most recent prediction, then request LLM insight for it.
   */
  const fetchInsight = useCallback(async () => {
    setLoading(true);
    setError('');
    setData(null);

    try {
      // 1. Get predictions list
      const predRes = await axios.get(`${API_BASE_URL}/predictions`);
      const predictions = predRes.data?.predictions;

      if (!predictions || predictions.length === 0) {
        setError('No predictions found. Upload an image or load demo data first.');
        setLoading(false);
        return;
      }

      const latest = predictions[0]; // most recent

      // 2. Call LLM insight endpoint
      const insightRes = await axios.post(`${API_BASE_URL}/llm-insight/${latest.id}`);
      setData(insightRes.data);
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Failed to fetch LLM insight.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  // ── NDVI color helper ──────────────────────────────────────────
  const ndviColor = (v) => {
    if (v >= 0.6) return 'text-green-600';
    if (v >= 0.4) return 'text-emerald-500';
    if (v >= 0.2) return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <div className="space-y-6">
      {/* ── Hero / Generate Button ─────────────────────────────── */}
      <div className="card text-center py-8">
        <div className="w-16 h-16 mx-auto bg-gradient-to-br from-purple-500 to-indigo-600 rounded-2xl flex items-center justify-center mb-4 shadow-lg">
          <Sparkles size={32} className="text-white" />
        </div>
        <h3 className="text-xl font-bold text-gray-800 dark:text-white">
          LLM-Powered Crop Insights
        </h3>
        <p className="text-gray-500 dark:text-gray-400 mt-1 max-w-md mx-auto">
          Analyse the latest prediction using an AI language model to generate
          detailed insights and actionable recommendations.
        </p>

        <button
          onClick={fetchInsight}
          disabled={loading}
          className="mt-6 inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white font-semibold rounded-xl shadow-md transition-all duration-200 disabled:opacity-60"
        >
          {loading ? (
            <>
              <Loader2 size={18} className="animate-spin" />
              Generating…
            </>
          ) : (
            <>
              <Sparkles size={18} />
              Generate Insight
            </>
          )}
        </button>
      </div>

      {/* ── Error ──────────────────────────────────────────────── */}
      {error && (
        <div className="flex items-start gap-3 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-700 rounded-xl text-red-700 dark:text-red-400 animate-fadeIn">
          <AlertCircle size={20} className="mt-0.5 flex-shrink-0" />
          <p>{error}</p>
        </div>
      )}

      {/* ── Results ────────────────────────────────────────────── */}
      {data && (
        <div className="space-y-6 animate-fadeIn">
          {/* Metric cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Crop Type */}
            <div className="card flex items-center gap-4">
              <div className="w-12 h-12 bg-orange-100 dark:bg-orange-900/30 rounded-xl flex items-center justify-center">
                <Leaf size={24} className="text-orange-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">Dominant Crop Type</p>
                <p className="text-lg font-bold text-gray-800 dark:text-white">{data.crop_type}</p>
              </div>
            </div>

            {/* NDVI */}
            <div className="card flex items-center gap-4">
              <div className="w-12 h-12 bg-emerald-100 dark:bg-emerald-900/30 rounded-xl flex items-center justify-center">
                <TrendingUp size={24} className="text-emerald-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">NDVI Value</p>
                <p className={`text-lg font-bold ${ndviColor(data.ndvi)}`}>
                  {data.ndvi.toFixed(2)}
                </p>
              </div>
            </div>

            {/* Yield */}
            <div className="card flex items-center gap-4">
              <div className="w-12 h-12 bg-amber-100 dark:bg-amber-900/30 rounded-xl flex items-center justify-center">
                <Wheat size={24} className="text-amber-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">Estimated Yield</p>
                <p className="text-lg font-bold text-gray-800 dark:text-white">
                  {data.estimated_yield_tons} <span className="text-sm font-normal text-gray-500">tons</span>
                </p>
              </div>
            </div>
          </div>

          {/* AI Insight */}
          <div className="card">
            <div className="flex items-center gap-2 mb-3">
              <Sparkles size={18} className="text-purple-600" />
              <h4 className="text-lg font-semibold text-gray-800 dark:text-white">AI Insight</h4>
            </div>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              {data.insight}
            </p>
          </div>

          {/* Recommendations */}
          <div className="card">
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle size={18} className="text-green-600" />
              <h4 className="text-lg font-semibold text-gray-800 dark:text-white">Recommendations</h4>
            </div>
            <ul className="space-y-2">
              {data.recommendations.map((rec, idx) => (
                <li key={idx} className="flex items-start gap-3">
                  <span className="mt-1.5 w-2 h-2 rounded-full bg-purple-500 flex-shrink-0" />
                  <span className="text-gray-700 dark:text-gray-300">{rec}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Model badge */}
          <div className="flex items-center justify-center gap-2 text-xs text-gray-400 dark:text-gray-500">
            <Sparkles size={12} />
            <span>Generated using Mistral-7B via Ollama</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default LLMInsights;
