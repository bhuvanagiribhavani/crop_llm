/**
 * ============================================================================
 * NDVI ANALYSIS COMPONENT
 * ============================================================================
 * Displays current NDVI value, color scale, vegetation classification,
 * and an interpretation guide based on segmentation statistics.
 */

import React from 'react';
import { AlertTriangle } from 'lucide-react';

const NDVIAnalysis = ({ statistics }) => {
  /**
   * Calculate a simulated NDVI value from land-cover percentages.
   * True NDVI requires NIR band; here we use a weighted proxy.
   */
  const calculateNDVI = () => {
    if (!statistics) return { value: 0.45, status: 'moderate' };

    const trees   = statistics['Trees']?.percentage || 0;
    const crops   = statistics['Crops']?.percentage || 0;
    const grass   = statistics['Grass']?.percentage || 0;
    const flooded = statistics['Flooded Vegetation']?.percentage || 0;

    // Weighted NDVI estimation
    const ndvi = (trees * 0.8 + crops * 0.65 + grass * 0.5 + flooded * 0.4) / 100;

    let status = 'sparse';
    if (ndvi > 0.6)      status = 'dense';
    else if (ndvi > 0.4) status = 'healthy';
    else if (ndvi > 0.2) status = 'moderate';

    return { value: Math.min(ndvi, 0.85), status };
  };

  const ndvi = calculateNDVI();

  const getNDVIColor = (v) => {
    if (v > 0.6) return 'text-green-600';
    if (v > 0.4) return 'text-emerald-500';
    if (v > 0.2) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getNDVIBgColor = (v) => {
    if (v > 0.6) return 'bg-green-500';
    if (v > 0.4) return 'bg-emerald-500';
    if (v > 0.2) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="space-y-6">
      {/* ── Current NDVI Value ───────────────────────────────────── */}
      <div className="card max-w-xl mx-auto">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">
          Current NDVI
        </h3>

        <div className="flex flex-col items-center py-6">
          {/* Large NDVI number */}
          <div className={`text-5xl font-bold ${getNDVIColor(ndvi.value)}`}>
            {ndvi.value.toFixed(2)}
          </div>
          <p className="text-gray-500 dark:text-gray-400 mt-2">
            Normalized Difference Vegetation Index
          </p>

          {/* ── NDVI Color Scale Bar ─────────────────────────────── */}
          <div className="w-full mt-6">
            <div className="h-4 rounded-full bg-gradient-to-r from-red-500 via-yellow-500 via-emerald-500 to-green-600 relative">
              <div
                className="absolute top-0 w-3 h-6 bg-white border-2 border-gray-800 rounded -mt-1"
                style={{ left: `${Math.min(ndvi.value * 100, 95)}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
              <span>0.0</span>
              <span>0.25</span>
              <span>0.5</span>
              <span>0.75</span>
              <span>1.0</span>
            </div>
          </div>

          {/* ── Vegetation Classification Badge ──────────────────── */}
          <div
            className={`mt-4 px-4 py-2 ${getNDVIBgColor(ndvi.value)} text-white rounded-full text-sm font-medium`}
          >
            {ndvi.status.charAt(0).toUpperCase() + ndvi.status.slice(1)} Vegetation
          </div>
        </div>
      </div>

      {/* ── NDVI Interpretation Guide ────────────────────────────── */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">
          NDVI Interpretation Guide
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            { range: '-1.0 to 0.0', label: 'Water / Non-vegetation', color: 'bg-blue-500',  desc: 'Water bodies, snow, clouds' },
            { range: '0.0 to 0.2',  label: 'Bare Soil / Sparse',     color: 'bg-amber-500', desc: 'Bare ground, rock, sand' },
            { range: '0.2 to 0.5',  label: 'Shrub / Grassland',      color: 'bg-lime-500',  desc: 'Moderate vegetation cover' },
            { range: '0.5 to 1.0',  label: 'Dense Vegetation',       color: 'bg-green-600', desc: 'Healthy crops, forests' },
          ].map((item, idx) => (
            <div key={idx} className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-xl">
              <div className={`w-full h-2 ${item.color} rounded-full mb-3`} />
              <p className="font-mono text-sm text-gray-600 dark:text-gray-300">{item.range}</p>
              <p className="font-semibold text-gray-800 dark:text-gray-100 mt-1">{item.label}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* ── RGB-only notice (when no statistics loaded) ───────────── */}
      {!statistics && (
        <div className="flex items-start gap-3 p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700 rounded-xl text-amber-700 dark:text-amber-400">
          <AlertTriangle size={20} className="mt-0.5" />
          <div>
            <p className="font-medium">Limited NDVI Data</p>
            <p className="text-sm mt-1">
              True NDVI calculation requires Near-Infrared (NIR) band data. The current
              analysis uses vegetation classification as a proxy for NDVI estimation.
              Upload an image for enhanced analysis.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default NDVIAnalysis;
