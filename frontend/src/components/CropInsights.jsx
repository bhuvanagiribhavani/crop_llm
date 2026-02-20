/**
 * ============================================================================
 * CROP INSIGHTS COMPONENT
 * ============================================================================
 * AI-powered insights and recommendations for crop analysis
 */

import React, { useState } from 'react';
import { 
  Brain, 
  Lightbulb, 
  TrendingUp, 
  AlertTriangle,
  CheckCircle,
  ChevronRight,
  Sparkles,
  FileText,
  Download,
  Share2
} from 'lucide-react';

const CropInsights = ({ insight, statistics }) => {
  const [expandedTip, setExpandedTip] = useState(null);

  // Generate recommendations based on statistics
  const generateRecommendations = () => {
    if (!statistics) return [];

    const recommendations = [];
    const cropsPercent = statistics['Crops']?.percentage || 0;
    const waterPercent = statistics['Water']?.percentage || 0;
    const bareGroundPercent = statistics['Bare Ground']?.percentage || 0;
    const vegetationPercent = (statistics['Trees']?.percentage || 0) + 
                              (statistics['Grass']?.percentage || 0) + 
                              (statistics['Scrub/Shrub']?.percentage || 0);

    if (cropsPercent > 30) {
      recommendations.push({
        type: 'success',
        title: 'Strong Agricultural Activity',
        description: 'The region shows excellent crop coverage. Consider crop rotation strategies for sustained yields.',
        icon: CheckCircle
      });
    } else if (cropsPercent < 10 && bareGroundPercent > 20) {
      recommendations.push({
        type: 'warning',
        title: 'Low Crop Coverage',
        description: 'Significant bare ground detected. This area may benefit from expanded agricultural development.',
        icon: AlertTriangle
      });
    }

    if (waterPercent < 2 && cropsPercent > 20) {
      recommendations.push({
        type: 'warning',
        title: 'Limited Water Resources',
        description: 'Low water body presence near crops. Consider implementing irrigation infrastructure.',
        icon: AlertTriangle
      });
    }

    if (vegetationPercent > 50) {
      recommendations.push({
        type: 'success',
        title: 'Healthy Ecosystem',
        description: 'Strong vegetation coverage indicates good soil health and biodiversity.',
        icon: CheckCircle
      });
    }

    return recommendations;
  };

  const recommendations = generateRecommendations();

  // Quick action tips
  const actionTips = [
    {
      title: 'Monitor Crop Health',
      description: 'Use NDVI analysis for vegetation health assessment',
      action: 'Run NDVI Analysis'
    },
    {
      title: 'Track Changes',
      description: 'Compare with historical data to detect land use changes',
      action: 'View History'
    },
    {
      title: 'Export Report',
      description: 'Generate comprehensive analysis report',
      action: 'Generate Report'
    }
  ];

  return (
    <div className="space-y-4">
      {/* AI Insight Card */}
      <div className="card bg-gradient-to-br from-primary-50 to-white border border-primary-100">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-primary-500 to-primary-700 rounded-xl flex items-center justify-center flex-shrink-0 shadow-lg">
            <Brain size={24} className="text-white" />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <h3 className="text-lg font-semibold text-gray-800">
                AI Analysis
              </h3>
              <Sparkles size={16} className="text-primary-500" />
            </div>
            <p className="text-gray-600 leading-relaxed">
              {insight || 'Upload a satellite image to receive AI-powered insights about crop coverage, vegetation health, and land use patterns.'}
            </p>
          </div>
        </div>
      </div>

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <div className="card">
          <div className="flex items-center gap-2 mb-4">
            <Lightbulb size={20} className="text-amber-500" />
            <h4 className="text-sm font-semibold text-gray-700">
              Recommendations
            </h4>
          </div>
          <div className="space-y-3">
            {recommendations.map((rec, idx) => {
              const Icon = rec.icon;
              const isWarning = rec.type === 'warning';
              return (
                <div 
                  key={idx}
                  className={`
                    p-4 rounded-xl border
                    ${isWarning 
                      ? 'bg-amber-50 border-amber-200' 
                      : 'bg-green-50 border-green-200'}
                  `}
                >
                  <div className="flex items-start gap-3">
                    <Icon 
                      size={20} 
                      className={isWarning ? 'text-amber-600' : 'text-green-600'} 
                    />
                    <div>
                      <p className={`font-medium ${isWarning ? 'text-amber-800' : 'text-green-800'}`}>
                        {rec.title}
                      </p>
                      <p className={`text-sm mt-1 ${isWarning ? 'text-amber-600' : 'text-green-600'}`}>
                        {rec.description}
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Quick Actions */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp size={20} className="text-primary-600" />
          <h4 className="text-sm font-semibold text-gray-700">
            Quick Actions
          </h4>
        </div>
        <div className="space-y-2">
          {actionTips.map((tip, idx) => (
            <button
              key={idx}
              onClick={() => setExpandedTip(expandedTip === idx ? null : idx)}
              className="w-full p-4 bg-gray-50 hover:bg-gray-100 rounded-xl transition-all duration-200 text-left group"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-gray-800">{tip.title}</p>
                  <p className="text-sm text-gray-500 mt-1">{tip.description}</p>
                </div>
                <ChevronRight 
                  size={20} 
                  className="text-gray-400 group-hover:text-primary-600 transition-colors" 
                />
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Export Options */}
      <div className="card">
        <h4 className="text-sm font-semibold text-gray-700 mb-4">
          Export & Share
        </h4>
        <div className="grid grid-cols-3 gap-3">
          <button className="flex flex-col items-center gap-2 p-4 bg-gray-50 hover:bg-gray-100 rounded-xl transition-colors">
            <FileText size={24} className="text-gray-600" />
            <span className="text-xs font-medium text-gray-600">PDF Report</span>
          </button>
          <button className="flex flex-col items-center gap-2 p-4 bg-gray-50 hover:bg-gray-100 rounded-xl transition-colors">
            <Download size={24} className="text-gray-600" />
            <span className="text-xs font-medium text-gray-600">Download</span>
          </button>
          <button className="flex flex-col items-center gap-2 p-4 bg-gray-50 hover:bg-gray-100 rounded-xl transition-colors">
            <Share2 size={24} className="text-gray-600" />
            <span className="text-xs font-medium text-gray-600">Share</span>
          </button>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="p-4 bg-gray-50 rounded-xl">
        <p className="text-xs text-gray-500 text-center">
          💡 Insights are generated using U-Net deep learning model trained on Sentinel-2 imagery.
          Results should be validated with ground truth data for critical applications.
        </p>
      </div>
    </div>
  );
};

export default CropInsights;
