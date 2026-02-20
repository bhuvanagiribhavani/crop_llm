/**
 * ============================================================================
 * CROP MAP PANEL COMPONENT
 * ============================================================================
 * Full-featured crop map view with quick actions and export options
 * 
 * Features:
 * - Tab-based image viewing (Input, Segmented, Insights)
 * - Quick action cards for navigation
 * - Export & Share functionality
 */

import React, { useState } from 'react';
import {
  Layers,
  Lightbulb,
  Image,
  Activity,
  History,
  FileText,
  Download,
  Share2,
  CheckCircle,
  AlertCircle,
  Copy,
  ExternalLink
} from 'lucide-react';

// Import components
import SegmentationView from './SegmentationView';
import CropInsights from './CropInsights';

// Import download utilities
import {
  downloadFullAnalysisPDF,
  downloadJSON,
  generateMockStatistics,
  generateMockInsight
} from '../utils/downloadHelpers';

const CropMapPanel = ({
  originalImage,
  segmentedImage,
  classesDetected,
  statistics,
  insight,
  onNavigate,  // Function to navigate to other pages
  setSuccess,  // Function to show success messages
}) => {
  // Local state
  const [activeTab, setActiveTab] = useState('segmented');
  const [notification, setNotification] = useState({ show: false, type: '', message: '' });

  // Tab configuration
  const tabs = [
    { id: 'input', label: 'Input Image', icon: Image },
    { id: 'segmented', label: 'Segmented Crop Map', icon: Layers },
    { id: 'insights', label: 'Crop Insights', icon: Lightbulb },
  ];

  /**
   * Show local notification toast
   */
  const showNotification = (type, message) => {
    setNotification({ show: true, type, message });
    setTimeout(() => {
      setNotification({ show: false, type: '', message: '' });
    }, 4000);
  };

  /**
   * Navigate to NDVI Analysis (Monitor Crop Health)
   */
  const handleMonitorCropHealth = () => {
    showNotification('success', '🌱 Opening NDVI-based crop health monitoring...');
    setTimeout(() => {
      if (onNavigate) onNavigate('ndvi');
    }, 800);
  };

  /**
   * Navigate to Statistics (Track Changes)
   */
  const handleTrackChanges = () => {
    showNotification('success', '📊 Tracking land-use changes using satellite history...');
    setTimeout(() => {
      if (onNavigate) onNavigate('statistics');
    }, 800);
  };

  /**
   * Navigate to Reports (Export Report)
   */
  const handleExportReport = () => {
    showNotification('success', '📄 Opening report export options...');
    setTimeout(() => {
      if (onNavigate) onNavigate('reports');
    }, 800);
  };

  /**
   * Download PDF Report for Crop Map
   */
  const handleDownloadPDFReport = () => {
    try {
      const reportData = {
        statistics: statistics || generateMockStatistics(),
        insight: insight || generateMockInsight(),
        classesDetected: classesDetected || ['Crops', 'Trees', 'Water', 'Grass']
      };

      const success = downloadFullAnalysisPDF('Crop_Map_Report.pdf', reportData);
      
      if (success) {
        showNotification('success', '✅ Crop Map Report downloaded successfully!');
      } else {
        showNotification('error', '❌ Failed to generate PDF report');
      }
    } catch (error) {
      showNotification('error', '❌ Error: ' + error.message);
    }
  };

  /**
   * Download crop data as JSON
   */
  const handleDownloadCropData = () => {
    try {
      const cropData = {
        analysis: {
          timestamp: new Date().toISOString(),
          imageProcessed: !!originalImage,
          segmentationComplete: !!segmentedImage
        },
        statistics: statistics || generateMockStatistics(),
        classesDetected: classesDetected || ['Crops', 'Trees', 'Water', 'Grass', 'Built Area'],
        metadata: {
          model: 'U-Net Segmentation',
          inputResolution: '256x256',
          outputClasses: 8
        }
      };

      const success = downloadJSON('crop_analysis_data.json', cropData);
      
      if (success) {
        showNotification('success', '✅ Crop data downloaded as JSON!');
      } else {
        showNotification('error', '❌ Failed to download data');
      }
    } catch (error) {
      showNotification('error', '❌ Error: ' + error.message);
    }
  };

  /**
   * Copy shareable link to clipboard
   */
  const handleCopyShareLink = async () => {
    try {
      // Generate a mock shareable link
      const shareLink = `https://crop-analytics.app/share/${Date.now().toString(36)}`;
      
      await navigator.clipboard.writeText(shareLink);
      showNotification('success', '🔗 Share link copied to clipboard!');
    } catch (error) {
      // Fallback for browsers without clipboard API
      const textArea = document.createElement('textarea');
      textArea.value = `https://crop-analytics.app/share/${Date.now().toString(36)}`;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      showNotification('success', '🔗 Share link copied to clipboard!');
    }
  };

  // Quick action cards configuration
  const quickActions = [
    {
      id: 'monitor',
      title: 'Monitor Crop Health',
      description: 'NDVI-based vegetation health analysis',
      icon: Activity,
      color: 'bg-emerald-500',
      hoverColor: 'hover:border-emerald-300',
      onClick: handleMonitorCropHealth
    },
    {
      id: 'track',
      title: 'Track Changes',
      description: 'Historical land-use comparison',
      icon: History,
      color: 'bg-blue-500',
      hoverColor: 'hover:border-blue-300',
      onClick: handleTrackChanges
    },
    {
      id: 'export',
      title: 'Export Report',
      description: 'Generate downloadable reports',
      icon: FileText,
      color: 'bg-purple-500',
      hoverColor: 'hover:border-purple-300',
      onClick: handleExportReport
    }
  ];

  return (
    <div className="space-y-6">
      {/* Notification Toast */}
      {notification.show && (
        <div className={`
          fixed top-4 right-4 z-50 flex items-center gap-3 px-4 py-3 rounded-xl shadow-lg
          animate-fadeIn transition-all duration-300
          ${notification.type === 'success' 
            ? 'bg-green-50 border border-green-200 text-green-700' 
            : 'bg-red-50 border border-red-200 text-red-700'}
        `}>
          {notification.type === 'success' ? (
            <CheckCircle size={20} />
          ) : (
            <AlertCircle size={20} />
          )}
          <span className="font-medium">{notification.message}</span>
        </div>
      )}

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <div className="flex gap-1">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`tab flex items-center gap-2 ${activeTab === tab.id ? 'active' : ''}`}
              >
                <Icon size={18} />
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Tab Content */}
      <div className="animate-fadeIn">
        {activeTab === 'input' && (
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Original Input Image</h3>
            {originalImage ? (
              <div className="flex justify-center">
                <img 
                  src={originalImage.startsWith('data:') ? originalImage : `data:image/png;base64,${originalImage}`}
                  alt="Original satellite"
                  className="max-w-md rounded-xl shadow-lg"
                />
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-64 text-gray-400">
                <Image size={48} className="mb-4 opacity-50" />
                <p className="font-medium">No image uploaded</p>
                <button
                  onClick={() => onNavigate && onNavigate('upload')}
                  className="btn-primary mt-4"
                >
                  Upload Image
                </button>
              </div>
            )}
          </div>
        )}
        
        {activeTab === 'segmented' && (
          <SegmentationView 
            originalImage={originalImage}
            segmentedImage={segmentedImage}
            classesDetected={classesDetected}
          />
        )}
        
        {activeTab === 'insights' && (
          <CropInsights 
            insight={insight}
            statistics={statistics}
          />
        )}
      </div>

      {/* Quick Actions Section */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Quick Actions</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {quickActions.map((action) => {
            const Icon = action.icon;
            return (
              <button
                key={action.id}
                onClick={action.onClick}
                className={`
                  p-4 border border-gray-200 rounded-xl text-left transition-all duration-200
                  hover:bg-gray-50 ${action.hoverColor} hover:shadow-md
                  group cursor-pointer
                `}
              >
                <div className="flex items-start gap-3">
                  <div className={`p-2 ${action.color} rounded-lg text-white group-hover:scale-110 transition-transform`}>
                    <Icon size={20} />
                  </div>
                  <div className="flex-1">
                    <h4 className="font-semibold text-gray-800 flex items-center gap-2">
                      {action.title}
                      <ExternalLink size={14} className="text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </h4>
                    <p className="text-sm text-gray-500 mt-1">{action.description}</p>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Export & Share Section */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Export & Share</h3>
        <div className="flex flex-wrap gap-3">
          {/* PDF Report Button */}
          <button
            onClick={handleDownloadPDFReport}
            className="flex items-center gap-2 px-4 py-2 bg-red-50 text-red-700 rounded-xl hover:bg-red-100 transition-colors border border-red-200"
          >
            <FileText size={18} />
            <span className="font-medium">PDF Report</span>
          </button>

          {/* Download Data Button */}
          <button
            onClick={handleDownloadCropData}
            className="flex items-center gap-2 px-4 py-2 bg-blue-50 text-blue-700 rounded-xl hover:bg-blue-100 transition-colors border border-blue-200"
          >
            <Download size={18} />
            <span className="font-medium">Download Data</span>
          </button>

          {/* Share Button */}
          <button
            onClick={handleCopyShareLink}
            className="flex items-center gap-2 px-4 py-2 bg-green-50 text-green-700 rounded-xl hover:bg-green-100 transition-colors border border-green-200"
          >
            <Share2 size={18} />
            <span className="font-medium">Share</span>
          </button>
        </div>

        {/* Share Info */}
        <div className="mt-4 p-3 bg-gray-50 rounded-xl">
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <Copy size={16} className="text-gray-400" />
            <span>Click Share to copy a shareable link to your clipboard</span>
          </div>
        </div>
      </div>

      {/* Analysis Summary */}
      {statistics && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Analysis Summary</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Crops', value: statistics['Crops']?.percentage || 0, color: 'bg-orange-500' },
              { label: 'Trees', value: statistics['Trees']?.percentage || 0, color: 'bg-green-600' },
              { label: 'Water', value: statistics['Water']?.percentage || 0, color: 'bg-blue-500' },
              { label: 'Built Area', value: statistics['Built Area']?.percentage || 0, color: 'bg-red-500' },
            ].map((item, idx) => (
              <div key={idx} className="text-center p-4 bg-gray-50 rounded-xl">
                <div className={`w-12 h-12 ${item.color} rounded-xl mx-auto mb-2 flex items-center justify-center text-white font-bold`}>
                  {item.value.toFixed(0)}%
                </div>
                <p className="text-sm font-medium text-gray-700">{item.label}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default CropMapPanel;
