/**
 * ============================================================================
 * CROP ANALYTICS DASHBOARD - MAIN APPLICATION
 * ============================================================================
 * Professional crop analytics dashboard using Sentinel-2 satellite imagery
 * 
 * Features:
 * - Upload satellite images
 * - View segmented crop maps
 * - Analyze crop statistics
 * - AI-powered insights
 * 
 * Author: Crop Analytics Project
 * Date: 2026
 * ============================================================================
 */

import React, { useState, useCallback, useEffect } from 'react';
import axios from 'axios';

// Components
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import UploadCard from './components/UploadCard';
import StatisticsPanel from './components/StatisticsPanel';
import NDVIAnalysis from './components/NDVIAnalysis';
import YieldEstimation from './components/YieldEstimation';
import ReportsPanel from './components/ReportsPanel';
import CropMapPanel from './components/CropMapPanel';
import LLMInsights from './components/LLMInsights';
import ChatBot from './components/ChatBot';

// Icons
import { 
  Layers, 
  Loader2,
  CheckCircle,
  AlertCircle,
  RefreshCw,
  Upload,
  Map,
  LineChart,
  Wheat
} from 'lucide-react';

// API Configuration - Use localhost for local access
const API_BASE_URL = 'http://localhost:5000';

function App() {
  // State management
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activePage, setActivePage] = useState('home');
  
  // Data states
  const [originalImage, setOriginalImage] = useState(null);
  const [segmentedImage, setSegmentedImage] = useState(null);
  const [statistics, setStatistics] = useState(null);
  const [insight, setInsight] = useState('');
  const [classesDetected, setClassesDetected] = useState([]);
  
  // Loading and error states
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  // Dark mode state
  const [darkMode, setDarkMode] = useState(() => {
    try {
      const stored = localStorage.getItem('crop_analytics_settings');
      return stored ? JSON.parse(stored).darkMode === true : false;
    } catch { return false; }
  });

  // Apply dark mode class to <html> element
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  /** Handle settings change from SettingsModal */
  const handleSettingsChange = useCallback((settings) => {
    setDarkMode(settings.darkMode);
  }, []);

  // Notification state
  const [notifications, setNotifications] = useState([]);

  /** Add a new notification */
  const addNotification = useCallback((title, message, type = 'prediction') => {
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    setNotifications((prev) => [
      {
        id: Date.now(),
        title,
        message,
        type,
        time: timeStr,
        read: false,
      },
      ...prev,
    ].slice(0, 20)); // keep last 20
  }, []);

  /** Clear all notifications */
  const clearNotifications = useCallback(() => {
    setNotifications([]);
  }, []);

  // Page titles and descriptions
  const pageConfig = {
    home: { title: 'Dashboard', description: 'Overview of your crop analytics' },
    upload: { title: 'Upload Image', description: 'Upload Sentinel-2 imagery for analysis' },
    cropmap: { title: 'Segmented Crop Map', description: 'View land cover classification results' },
    ndvi: { title: 'NDVI Analysis', description: 'Vegetation index analysis and trends' },
    statistics: { title: 'Crop Statistics', description: 'Detailed statistical breakdown' },
    yield: { title: 'Yield Estimation', description: 'AI-powered crop yield predictions' },
    'llm-insights': { title: 'LLM Insights', description: 'AI-generated crop analysis and recommendations' },
    reports: { title: 'Reports', description: 'Generate and download reports' },
    help: { title: 'Help & Support', description: 'Get help and documentation' },
  };

  // Get current page config
  const currentPage = pageConfig[activePage] || pageConfig.home;

  /**
   * Handle image upload and prediction
   */
  const handleImageUpload = useCallback(async (file) => {
    setIsLoading(true);
    setError('');
    setSuccess('');

    console.log('Starting upload to:', API_BASE_URL);
    console.log('File:', file.name, file.size, 'bytes');

    try {
      // Create form data
      const formData = new FormData();
      formData.append('image', file);

      console.log('Sending request...');
      
      // Send to backend API
      const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // 60 second timeout
      });

      if (response.data.success) {
        // Update state with results
        setOriginalImage(response.data.original_base64);
        setSegmentedImage(response.data.mask_base64);
        setStatistics(response.data.statistics);
        setInsight(response.data.insight);
        setClassesDetected(response.data.classes_detected || []);
        
        setSuccess('Analysis complete! View the segmented map below.');
        addNotification('Crop prediction saved', `Detected ${response.data.classes_detected?.length || 0} land cover classes`, 'prediction');
        // Auto-navigate to cropmap page after successful upload
        setActivePage('cropmap');
      } else {
        // Clear previous results on invalid image
        setOriginalImage(null);
        setSegmentedImage(null);
        setStatistics(null);
        setInsight('');
        setClassesDetected([]);
        setError(response.data.error || 'Prediction failed');
      }
    } catch (err) {
      console.error('Upload error:', err);
      
      // Clear previous results on error
      setOriginalImage(null);
      setSegmentedImage(null);
      setStatistics(null);
      setInsight('');
      setClassesDetected([]);
      
      if (err.code === 'ECONNABORTED') {
        setError('Request timed out. Please try again.');
      } else if (err.response) {
        // Handle 400 error (invalid image) specifically
        const errorMsg = err.response.data?.error || 'Server error occurred';
        setError(errorMsg);
      } else if (err.request) {
        setError('Cannot connect to server. Make sure the backend is running on port 5000.');
      } else {
        setError('An unexpected error occurred');
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Load demo data for testing
   */
  const handleLoadDemo = useCallback(async () => {
    setIsLoading(true);
    setError('');
    
    try {
      const response = await axios.get(`${API_BASE_URL}/predict/demo`);
      
      if (response.data.success) {
        setStatistics(response.data.statistics);
        setInsight(response.data.insight);
        setClassesDetected(response.data.classes_detected || []);
        setSuccess('Demo data loaded successfully!');
        addNotification('Demo data loaded', 'Sample crop statistics are ready to view', 'analysis');
        setActivePage('statistics');
      }
    } catch (err) {
      setError('Failed to load demo data. Make sure the backend is running.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Reset all data
   */
  const handleReset = useCallback(() => {
    setOriginalImage(null);
    setSegmentedImage(null);
    setStatistics(null);
    setInsight('');
    setClassesDetected([]);
    setError('');
    setSuccess('');
  }, []);

  return (
    <div className="flex h-screen bg-primary-50/30 dark:bg-gray-900 transition-colors duration-300">
      {/* Sidebar Navigation */}
      <Sidebar 
        activePage={activePage}
        setActivePage={setActivePage}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden transition-colors duration-300">
        {/* Header */}
        <Header
          onMenuClick={() => setSidebarOpen(true)}
          onNavigate={setActivePage}
          notifications={notifications}
          onClearNotifications={clearNotifications}
          onSettingsChange={handleSettingsChange}
        />

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto p-6 dark:bg-gray-900 transition-colors duration-300">
          <div className="max-w-7xl mx-auto space-y-6">
            
            {/* Page Title */}
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold text-gray-800 dark:text-white">
                  {currentPage.title}
                </h2>
                <p className="text-gray-500 dark:text-gray-400 mt-1">
                  {currentPage.description}
                </p>
              </div>
              
              {/* Action Buttons */}
              <div className="flex items-center gap-3">
                <button
                  onClick={handleLoadDemo}
                  className="btn-secondary"
                  disabled={isLoading}
                >
                  Load Demo
                </button>
                {(originalImage || statistics) && (
                  <button
                    onClick={handleReset}
                    className="btn-secondary flex items-center gap-2"
                  >
                    <RefreshCw size={16} />
                    Reset
                  </button>
                )}
              </div>
            </div>

            {/* Status Messages */}
            {error && (
              <div className={`flex items-start gap-3 p-4 rounded-xl animate-fadeIn ${
                error.includes('Invalid input image') 
                  ? 'bg-red-100 border-2 border-red-400 text-red-800' 
                  : 'bg-red-50 border border-red-200 text-red-700'
              }`}>
                <AlertCircle size={24} className={`flex-shrink-0 mt-0.5 ${
                  error.includes('Invalid input image') ? 'text-red-600' : ''
                }`} />
                <div className="flex-1">
                  <p className={`${error.includes('Invalid input image') ? 'font-semibold' : ''}`}>
                    {error}
                  </p>
                  {error.includes('Invalid input image') && (
                    <p className="text-sm mt-2 text-red-600">
                      💡 Tip: Upload RGB satellite imagery from Sentinel-2 for accurate land cover segmentation.
                    </p>
                  )}
                </div>
              </div>
            )}
            
            {success && (
              <div className="flex items-center gap-3 p-4 bg-green-50 border border-green-200 rounded-xl text-green-700 animate-fadeIn">
                <CheckCircle size={20} />
                <span>{success}</span>
              </div>
            )}

            {/* ============================================================ */}
            {/* PAGE CONTENT - Renders based on activePage state */}
            {/* ============================================================ */}

            {/* HOME / DASHBOARD PAGE */}
            {activePage === 'home' && (
              <div className="animate-fadeIn space-y-6">
                {/* Upload Section */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div className="lg:col-span-1">
                    <UploadCard 
                      onImageUpload={handleImageUpload}
                      isLoading={isLoading}
                    />
                  </div>

                  {/* Quick Stats Preview */}
                  <div className="lg:col-span-2">
                    <div className="card h-full">
                      <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">
                        Analysis Overview
                      </h3>
                      
                      {isLoading ? (
                        <div className="flex flex-col items-center justify-center h-48">
                          <Loader2 size={40} className="text-primary-600 animate-spin mb-4" />
                          <p className="text-gray-600 dark:text-gray-300">Processing image...</p>
                          <p className="text-sm text-gray-400 mt-1">Running U-Net segmentation model</p>
                        </div>
                      ) : statistics ? (
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          {[
                            { label: 'Crops', value: statistics['Crops']?.percentage || 0, color: 'bg-orange-500' },
                            { label: 'Trees', value: statistics['Trees']?.percentage || 0, color: 'bg-green-600' },
                            { label: 'Water', value: statistics['Water']?.percentage || 0, color: 'bg-blue-500' },
                            { label: 'Built Area', value: statistics['Built Area']?.percentage || 0, color: 'bg-red-500' },
                          ].map((item, idx) => (
                            <div key={idx} className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-xl cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors" onClick={() => setActivePage('statistics')}>
                              <div className={`w-12 h-12 ${item.color} rounded-xl mx-auto mb-2 flex items-center justify-center text-white font-bold`}>
                                {item.value.toFixed(0)}%
                              </div>
                              <p className="text-sm font-medium text-gray-700 dark:text-gray-200">{item.label}</p>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="flex flex-col items-center justify-center h-48 text-gray-400">
                          <Layers size={48} className="mb-4 opacity-50" />
                          <p className="font-medium">No analysis yet</p>
                          <p className="text-sm">Upload an image to see results</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Quick Navigation Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {[
                    { id: 'upload', icon: Upload, label: 'Upload Image', desc: 'Add new satellite imagery', color: 'bg-blue-500' },
                    { id: 'cropmap', icon: Map, label: 'Crop Map', desc: 'View segmentation results', color: 'bg-green-500' },
                    { id: 'ndvi', icon: LineChart, label: 'NDVI Analysis', desc: 'Vegetation health index', color: 'bg-emerald-500' },
                    { id: 'yield', icon: Wheat, label: 'Yield Estimation', desc: 'Predict crop yields', color: 'bg-amber-500' },
                  ].map((item) => {
                    const Icon = item.icon;
                    return (
                      <button
                        key={item.id}
                        onClick={() => setActivePage(item.id)}
                        className="card hover:shadow-lg transition-all duration-300 text-left group"
                      >
                        <div className={`w-12 h-12 ${item.color} rounded-xl mb-3 flex items-center justify-center text-white group-hover:scale-110 transition-transform`}>
                          <Icon size={24} />
                        </div>
                        <h4 className="font-semibold text-gray-800 dark:text-white">{item.label}</h4>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{item.desc}</p>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* UPLOAD PAGE */}
            {activePage === 'upload' && (
              <div className="animate-fadeIn">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <UploadCard 
                    onImageUpload={handleImageUpload}
                    isLoading={isLoading}
                  />
                  
                  <div className="card">
                    <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">Upload Guidelines</h3>
                    <ul className="space-y-3 text-gray-600 dark:text-gray-300">
                      <li className="flex items-start gap-3">
                        <CheckCircle size={18} className="text-green-500 mt-0.5" />
                        <span>Supported formats: PNG, JPG, TIFF</span>
                      </li>
                      <li className="flex items-start gap-3">
                        <CheckCircle size={18} className="text-green-500 mt-0.5" />
                        <span>Recommended resolution: 64x64 to 256x256 pixels</span>
                      </li>
                      <li className="flex items-start gap-3">
                        <CheckCircle size={18} className="text-green-500 mt-0.5" />
                        <span>RGB satellite imagery (Sentinel-2 recommended)</span>
                      </li>
                      <li className="flex items-start gap-3">
                        <CheckCircle size={18} className="text-green-500 mt-0.5" />
                        <span>Maximum file size: 10MB</span>
                      </li>
                    </ul>
                    
                    {originalImage && (
                      <div className="mt-6 pt-6 border-t border-gray-100 dark:border-gray-700">
                        <p className="text-sm text-green-600 font-medium flex items-center gap-2">
                          <CheckCircle size={16} />
                          Image uploaded successfully! View results in Crop Map.
                        </p>
                        <button
                          onClick={() => setActivePage('cropmap')}
                          className="btn-primary mt-3"
                        >
                          View Segmentation Results
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* CROP MAP PAGE (with tabs, quick actions, export) */}
            {activePage === 'cropmap' && (
              <div className="animate-fadeIn">
                <CropMapPanel
                  originalImage={originalImage}
                  segmentedImage={segmentedImage}
                  classesDetected={classesDetected}
                  statistics={statistics}
                  insight={insight}
                  onNavigate={setActivePage}
                  setSuccess={setSuccess}
                />
              </div>
            )}

            {/* NDVI ANALYSIS PAGE */}
            {activePage === 'ndvi' && (
              <div className="animate-fadeIn">
                <NDVIAnalysis statistics={statistics} />
              </div>
            )}

            {/* STATISTICS PAGE */}
            {activePage === 'statistics' && (
              <div className="animate-fadeIn">
                <StatisticsPanel statistics={statistics} />
              </div>
            )}

            {/* YIELD ESTIMATION PAGE */}
            {activePage === 'yield' && (
              <div className="animate-fadeIn">
                <YieldEstimation statistics={statistics} insight={insight} />
              </div>
            )}

            {/* LLM INSIGHTS PAGE */}
            {activePage === 'llm-insights' && (
              <div className="animate-fadeIn">
                <LLMInsights />
              </div>
            )}

            {/* REPORTS PAGE */}
            {activePage === 'reports' && (
              <div className="animate-fadeIn">
                <ReportsPanel
                  statistics={statistics}
                  insight={insight}
                  classesDetected={classesDetected}
                  onLoadDemo={handleLoadDemo}
                  onReset={handleReset}
                  isLoading={isLoading}
                />
              </div>
            )}

            {/* HELP PAGE */}
            {activePage === 'help' && (
              <div className="animate-fadeIn">
                <div className="card">
<h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">Help & Support</h3>
                  
                  <div className="space-y-6">
                    <div>
                      <h4 className="font-semibold text-gray-700 dark:text-gray-200 mb-2">Getting Started</h4>
                      <p className="text-gray-600 dark:text-gray-400">Upload a Sentinel-2 satellite image to begin analysis. The U-Net model will automatically segment the image into 8 land cover classes.</p>
                    </div>
                    
                    <div>
                      <h4 className="font-semibold text-gray-700 dark:text-gray-200 mb-2">Land Cover Classes</h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                        {['Water', 'Trees', 'Grass', 'Flooded Vegetation', 'Crops', 'Scrub/Shrub', 'Built Area', 'Bare Ground'].map((cls, idx) => (
                          <span key={idx} className="px-3 py-2 bg-gray-100 dark:bg-gray-700 rounded-lg text-sm text-gray-700 dark:text-gray-200">{cls}</span>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="font-semibold text-gray-700 dark:text-gray-200 mb-2">Contact Support</h4>
                      <p className="text-gray-600 dark:text-gray-400">For technical assistance, please contact the development team.</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

          </div>
        </main>

        {/* Footer */}
        <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 px-6 py-3 transition-colors duration-300">
          <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>© 2026 Crop Analytics Dashboard | Powered by U-Net Deep Learning</span>
            <span>Sentinel-2 Satellite Imagery Analysis</span>
          </div>
        </footer>
      </div>

      {/* Floating Help Chatbot */}
      <ChatBot />
    </div>
  );
}

export default App;
