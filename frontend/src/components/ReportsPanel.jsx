/**
 * ============================================================================
 * REPORTS PANEL COMPONENT
 * ============================================================================
 * Full-featured reports generation and download panel
 * 
 * Features:
 * - PDF report generation
 * - CSV/JSON data export
 * - Excel time series export
 * - Demo data loading
 * - Toast notifications
 */

import React, { useState } from 'react';
import {
  FileText,
  Download,
  FileJson,
  Table,
  FileSpreadsheet,
  Leaf,
  CheckCircle,
  AlertCircle,
  Loader2,
  Save,
  RefreshCw
} from 'lucide-react';

// Import download utilities
import {
  downloadFullAnalysisPDF,
  downloadCropHealthSummaryPDF,
  downloadCSV,
  downloadJSON,
  downloadExcel,
  generateMockNDVIData,
  generateMockStatistics,
  generateMockInsight
} from '../utils/downloadHelpers';

const ReportsPanel = ({ 
  statistics, 
  insight, 
  classesDetected,
  onLoadDemo,
  onReset,
  isLoading 
}) => {
  // Local state for notifications
  const [notification, setNotification] = useState({ show: false, type: '', message: '' });
  const [downloadingItem, setDownloadingItem] = useState(null);

  /**
   * Show notification message
   * @param {string} type - 'success' or 'error'
   * @param {string} message - Notification message
   */
  const showNotification = (type, message) => {
    setNotification({ show: true, type, message });
    // Auto-hide after 4 seconds
    setTimeout(() => {
      setNotification({ show: false, type: '', message: '' });
    }, 4000);
  };

  /**
   * Handle Full Analysis Report PDF download
   */
  const handleFullAnalysisReport = async () => {
    setDownloadingItem('full-analysis');
    
    try {
      // Use actual statistics or mock data
      const reportData = {
        statistics: statistics || generateMockStatistics(),
        insight: insight || generateMockInsight(),
        classesDetected: classesDetected || ['Crops', 'Trees', 'Water', 'Built Area']
      };

      const success = downloadFullAnalysisPDF(
        'Full_Analysis_Report.pdf',
        reportData
      );

      if (success) {
        showNotification('success', '✅ Full analysis report downloaded successfully!');
      } else {
        showNotification('error', '❌ Failed to generate PDF report');
      }
    } catch (error) {
      showNotification('error', '❌ Error generating report: ' + error.message);
    } finally {
      setDownloadingItem(null);
    }
  };

  /**
   * Handle Crop Health Summary PDF download
   */
  const handleCropHealthSummary = async () => {
    setDownloadingItem('crop-health');
    
    try {
      const reportData = {
        statistics: statistics || generateMockStatistics(),
        insight: insight || generateMockInsight(),
        classesDetected: classesDetected || ['Crops', 'Trees', 'Grass']
      };

      const success = downloadCropHealthSummaryPDF(
        'Crop_Health_Summary.pdf',
        reportData
      );

      if (success) {
        showNotification('success', '✅ Crop health summary downloaded successfully!');
      } else {
        showNotification('error', '❌ Failed to generate health summary');
      }
    } catch (error) {
      showNotification('error', '❌ Error generating summary: ' + error.message);
    } finally {
      setDownloadingItem(null);
    }
  };

  /**
   * Handle Statistics Export (CSV + JSON)
   */
  const handleExportStatistics = async () => {
    setDownloadingItem('export-stats');
    
    try {
      const statsData = {
        statistics: statistics || generateMockStatistics()
      };

      // Download both CSV and JSON
      const csvSuccess = downloadCSV('statistics.csv', statsData);
      
      // Small delay between downloads
      await new Promise(resolve => setTimeout(resolve, 500));
      
      const jsonSuccess = downloadJSON('statistics.json', statsData);

      if (csvSuccess && jsonSuccess) {
        showNotification('success', '✅ Statistics exported as CSV and JSON!');
      } else {
        showNotification('error', '❌ Some files failed to download');
      }
    } catch (error) {
      showNotification('error', '❌ Error exporting statistics: ' + error.message);
    } finally {
      setDownloadingItem(null);
    }
  };

  /**
   * Handle NDVI Time Series Excel download
   */
  const handleNDVITimeSeries = async () => {
    setDownloadingItem('ndvi-series');
    
    try {
      const ndviData = {
        ndviTimeSeries: generateMockNDVIData()
      };

      const success = downloadExcel('ndvi_timeseries.xls', ndviData);

      if (success) {
        showNotification('success', '✅ NDVI time series downloaded as Excel!');
      } else {
        showNotification('error', '❌ Failed to generate Excel file');
      }
    } catch (error) {
      showNotification('error', '❌ Error generating Excel: ' + error.message);
    } finally {
      setDownloadingItem(null);
    }
  };

  /**
   * Handle Save All Reports (combined download)
   */
  const handleSaveAllReports = async () => {
    setDownloadingItem('save-all');
    
    try {
      const allData = {
        metadata: {
          generated: new Date().toISOString(),
          source: 'Crop Analytics Dashboard',
          version: '1.0.0'
        },
        statistics: statistics || generateMockStatistics(),
        insight: insight || generateMockInsight(),
        classesDetected: classesDetected || ['Crops', 'Trees', 'Water', 'Grass', 'Built Area'],
        ndviTimeSeries: generateMockNDVIData()
      };

      // Download comprehensive JSON report
      const success = downloadJSON('Complete_Crop_Report.json', allData);

      if (success) {
        showNotification('success', '✅ Complete report saved successfully!');
      } else {
        showNotification('error', '❌ Failed to save report');
      }
    } catch (error) {
      showNotification('error', '❌ Error saving report: ' + error.message);
    } finally {
      setDownloadingItem(null);
    }
  };

  /**
   * Handle Load Demo Data
   */
  const handleLoadDemoData = () => {
    if (onLoadDemo) {
      onLoadDemo();
    }
    showNotification('success', '✅ Demo data loaded successfully!');
  };

  /**
   * Handle Reset Data
   */
  const handleResetData = () => {
    if (onReset) {
      onReset();
    }
    showNotification('success', '🔄 Data reset successfully!');
  };

  // Report card configuration
  const reportCards = [
    {
      id: 'full-analysis',
      title: 'Full Analysis Report',
      description: 'Complete segmentation and statistics report (PDF)',
      icon: FileText,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
      borderColor: 'hover:border-blue-300',
      onClick: handleFullAnalysisReport
    },
    {
      id: 'crop-health',
      title: 'Crop Health Summary',
      description: 'Agricultural insights summary (PDF)',
      icon: Leaf,
      color: 'text-green-600',
      bgColor: 'bg-green-100',
      borderColor: 'hover:border-green-300',
      onClick: handleCropHealthSummary
    },
    {
      id: 'export-stats',
      title: 'Export Statistics',
      description: 'Download raw data (CSV/JSON)',
      icon: Table,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
      borderColor: 'hover:border-purple-300',
      onClick: handleExportStatistics
    },
    {
      id: 'ndvi-series',
      title: 'NDVI Time Series',
      description: 'Historical vegetation analysis (Excel)',
      icon: FileSpreadsheet,
      color: 'text-emerald-600',
      bgColor: 'bg-emerald-100',
      borderColor: 'hover:border-emerald-300',
      onClick: handleNDVITimeSeries
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

      {/* Main Card */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold text-gray-800">Reports</h3>
            <p className="text-gray-600">Generate and download analysis reports</p>
          </div>
          
          {/* Action Buttons */}
          <div className="flex items-center gap-2">
            <button
              onClick={handleLoadDemoData}
              disabled={isLoading}
              className="btn-secondary flex items-center gap-2 text-sm"
            >
              {isLoading ? <Loader2 size={16} className="animate-spin" /> : <Download size={16} />}
              Load Demo
            </button>
            <button
              onClick={handleResetData}
              className="btn-secondary flex items-center gap-2 text-sm"
            >
              <RefreshCw size={16} />
              Reset
            </button>
          </div>
        </div>
        
        {/* Report Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {reportCards.map((card) => {
            const Icon = card.icon;
            const isDownloading = downloadingItem === card.id;
            
            return (
              <button
                key={card.id}
                onClick={card.onClick}
                disabled={isDownloading}
                className={`
                  p-4 border border-gray-200 rounded-xl text-left transition-all duration-200
                  hover:bg-gray-50 ${card.borderColor} hover:shadow-md
                  ${isDownloading ? 'opacity-75 cursor-wait' : 'cursor-pointer'}
                  group
                `}
              >
                <div className="flex items-start gap-3">
                  <div className={`p-2 ${card.bgColor} rounded-lg group-hover:scale-110 transition-transform`}>
                    {isDownloading ? (
                      <Loader2 size={20} className={`${card.color} animate-spin`} />
                    ) : (
                      <Icon size={20} className={card.color} />
                    )}
                  </div>
                  <div className="flex-1">
                    <h4 className="font-semibold text-gray-800 flex items-center gap-2">
                      {card.title}
                      <Download size={14} className="text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </h4>
                    <p className="text-sm text-gray-500 mt-1">{card.description}</p>
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Save All Reports Button */}
        <div className="mt-6 pt-6 border-t border-gray-100">
          <button
            onClick={handleSaveAllReports}
            disabled={downloadingItem === 'save-all'}
            className="w-full btn-primary flex items-center justify-center gap-2"
          >
            {downloadingItem === 'save-all' ? (
              <>
                <Loader2 size={18} className="animate-spin" />
                Generating Report...
              </>
            ) : (
              <>
                <Save size={18} />
                Save Complete Report
              </>
            )}
          </button>
          <p className="text-xs text-gray-500 text-center mt-2">
            Downloads all analysis data as a single JSON file
          </p>
        </div>
      </div>

      {/* Data Status Card */}
      <div className="card">
        <h4 className="font-semibold text-gray-800 mb-4">Report Data Status</h4>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className={`p-3 rounded-lg ${statistics ? 'bg-green-50 border border-green-200' : 'bg-gray-50 border border-gray-200'}`}>
            <div className="flex items-center gap-2">
              {statistics ? (
                <CheckCircle size={18} className="text-green-600" />
              ) : (
                <AlertCircle size={18} className="text-gray-400" />
              )}
              <span className={`font-medium ${statistics ? 'text-green-700' : 'text-gray-600'}`}>
                Statistics
              </span>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              {statistics ? `${Object.keys(statistics).length} classes detected` : 'No data loaded'}
            </p>
          </div>

          <div className={`p-3 rounded-lg ${insight ? 'bg-green-50 border border-green-200' : 'bg-gray-50 border border-gray-200'}`}>
            <div className="flex items-center gap-2">
              {insight ? (
                <CheckCircle size={18} className="text-green-600" />
              ) : (
                <AlertCircle size={18} className="text-gray-400" />
              )}
              <span className={`font-medium ${insight ? 'text-green-700' : 'text-gray-600'}`}>
                AI Insights
              </span>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              {insight ? 'Analysis available' : 'No insights generated'}
            </p>
          </div>

          <div className="p-3 rounded-lg bg-blue-50 border border-blue-200">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <span className="font-medium text-blue-700">System Ready</span>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Model loaded & ready
            </p>
          </div>
        </div>
      </div>

      {/* Quick Format Reference */}
      <div className="card">
        <h4 className="font-semibold text-gray-800 mb-4">Export Format Reference</h4>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { format: 'PDF', icon: FileText, desc: 'Printable reports', color: 'text-red-500' },
            { format: 'CSV', icon: Table, desc: 'Spreadsheet data', color: 'text-green-500' },
            { format: 'JSON', icon: FileJson, desc: 'Structured data', color: 'text-yellow-500' },
            { format: 'Excel', icon: FileSpreadsheet, desc: 'Time series', color: 'text-blue-500' },
          ].map((item, idx) => {
            const Icon = item.icon;
            return (
              <div key={idx} className="flex items-center gap-2 p-2 bg-gray-50 rounded-lg">
                <Icon size={18} className={item.color} />
                <div>
                  <p className="font-medium text-gray-800 text-sm">{item.format}</p>
                  <p className="text-xs text-gray-500">{item.desc}</p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default ReportsPanel;
