/**
 * ============================================================================
 * YIELD ESTIMATION COMPONENT
 * ============================================================================
 * Displays estimated crop yield (in tons) based on segmentation analysis.
 */

import React from 'react';
import {
  Wheat,
  TrendingUp,
  MapPin,
  AlertCircle,
  CheckCircle
} from 'lucide-react';

const YieldEstimation = ({ statistics }) => {
  /**
   * Calculate yield estimates.
   * Yield is calculated and displayed in tons.
   */
  const calculateYieldEstimate = () => {
    if (!statistics) {
      return {
        totalArea: 100,
        cropArea: 0,
        estimatedYield: 0,
        yieldPerHectare: 0,
        confidence: 0
      };
    }

    const cropPercentage  = statistics['Crops']?.percentage || 0;
    const grassPercentage = statistics['Grass']?.percentage || 0;
    const waterPercentage = statistics['Water']?.percentage || 0;

    const totalArea = 100; // hectares (simulated)
    const cropArea  = (cropPercentage / 100) * totalArea;
    const baseYieldTons = 4.5; // tons per hectare

    let yieldMultiplier = 1.0;
    if (waterPercentage > 5 && waterPercentage < 20) yieldMultiplier += 0.15;
    if (grassPercentage > 10) yieldMultiplier -= 0.1;

    const yieldPerHectareTons = baseYieldTons * yieldMultiplier;
    const estimatedTons       = cropArea * yieldPerHectareTons;

    const estimatedYield    = estimatedTons;
    const yieldPerHectare   = yieldPerHectareTons;

    const confidence = Math.min(
      95,
      60 + (cropPercentage > 10 ? 25 : 0) + (cropPercentage > 30 ? 10 : 0)
    );

    return {
      totalArea,
      cropArea: cropArea.toFixed(1),
      estimatedYield: estimatedYield.toFixed(1),
      yieldPerHectare: yieldPerHectare.toFixed(2),
      confidence
    };
  };

  const yieldData = calculateYieldEstimate();

  return (
    <div className="space-y-6">
      {/* ── Yield Metric Cards ────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Estimated Yield (tons) */}
        <div className="card bg-gradient-to-br from-amber-500 to-amber-600 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-amber-100 text-sm">Estimated Yield</p>
              <p className="text-3xl font-bold mt-1">{yieldData.estimatedYield}</p>
              <p className="text-amber-100 text-sm mt-1">tons</p>
            </div>
            <Wheat size={40} className="text-amber-200" />
          </div>
        </div>

        {/* Crop Area */}
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-500 dark:text-gray-400 text-sm">Crop Area</p>
              <p className="text-2xl font-bold text-gray-800 dark:text-white mt-1">{yieldData.cropArea}</p>
              <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">hectares</p>
            </div>
            <MapPin size={32} className="text-green-500" />
          </div>
        </div>

        {/* Yield per Hectare (tons/ha) */}
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-500 dark:text-gray-400 text-sm">Yield per Hectare</p>
              <p className="text-2xl font-bold text-gray-800 dark:text-white mt-1">{yieldData.yieldPerHectare}</p>
              <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">tons/ha</p>
            </div>
            <TrendingUp size={32} className="text-emerald-500" />
          </div>
        </div>

        {/* Confidence */}
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-500 dark:text-gray-400 text-sm">Confidence</p>
              <p className="text-2xl font-bold text-gray-800 dark:text-white mt-1">{yieldData.confidence}%</p>
              <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">accuracy</p>
            </div>
            <div className={`w-12 h-12 rounded-full flex items-center justify-center ${yieldData.confidence > 70 ? 'bg-green-100 dark:bg-green-900/30' : 'bg-amber-100 dark:bg-amber-900/30'}`}>
              {yieldData.confidence > 70 ? (
                <CheckCircle size={24} className="text-green-600" />
              ) : (
                <AlertCircle size={24} className="text-amber-600" />
              )}
            </div>
          </div>
        </div>
      </div>

      {/* ── No-data notice ────────────────────────────────────────── */}
      {!statistics && (
        <div className="flex items-start gap-3 p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700 rounded-xl text-amber-700 dark:text-amber-400">
          <AlertCircle size={20} className="mt-0.5" />
          <div>
            <p className="font-medium">No Image Analyzed</p>
            <p className="text-sm mt-1">
              Upload a satellite image to get accurate yield estimations based on
              actual crop coverage data. Current values are based on sample projections.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default YieldEstimation;
