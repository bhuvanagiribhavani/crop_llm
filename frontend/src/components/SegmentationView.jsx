/**
 * ============================================================================
 * SEGMENTATION VIEW COMPONENT
 * ============================================================================
 * Display input image and segmented crop map side by side
 */

import React, { useState } from 'react';
import { 
  Image as ImageIcon, 
  Layers, 
  ZoomIn, 
  ZoomOut,
  Maximize2,
  Download,
  Eye,
  EyeOff
} from 'lucide-react';

const SegmentationView = ({ originalImage, segmentedImage, classesDetected }) => {
  const [activeTab, setActiveTab] = useState('segmented');
  const [showLegend, setShowLegend] = useState(true);
  const [zoom, setZoom] = useState(1);

  // Class color mapping
  const classColors = {
    'Water': { color: '#419BDF', description: 'Rivers, lakes, ponds' },
    'Trees': { color: '#397D49', description: 'Forests, orchards' },
    'Grass': { color: '#88B053', description: 'Pastures, lawns' },
    'Flooded Vegetation': { color: '#7A87C6', description: 'Rice paddies, wetlands' },
    'Crops': { color: '#E4963A', description: 'Agricultural fields' },
    'Scrub/Shrub': { color: '#DFC17D', description: 'Bushes, shrubland' },
    'Built Area': { color: '#C4281B', description: 'Urban areas, buildings' },
    'Bare Ground': { color: '#A59B8F', description: 'Exposed soil, sand' },
  };

  const tabs = [
    { id: 'input', label: 'Input Image', icon: ImageIcon },
    { id: 'segmented', label: 'Segmented Map', icon: Layers },
  ];

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.25, 3));
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.25, 0.5));

  return (
    <div className="card">
      {/* Header with Tabs */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex gap-1 bg-gray-100 p-1 rounded-xl">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium
                  transition-all duration-200
                  ${activeTab === tab.id 
                    ? 'bg-white text-primary-700 shadow-sm' 
                    : 'text-gray-500 hover:text-gray-700'}
                `}
              >
                <Icon size={16} />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          <button 
            onClick={handleZoomOut}
            className="p-2 hover:bg-gray-100 rounded-lg text-gray-500"
          >
            <ZoomOut size={18} />
          </button>
          <span className="text-sm text-gray-500 min-w-[50px] text-center">
            {Math.round(zoom * 100)}%
          </span>
          <button 
            onClick={handleZoomIn}
            className="p-2 hover:bg-gray-100 rounded-lg text-gray-500"
          >
            <ZoomIn size={18} />
          </button>
          <div className="w-px h-6 bg-gray-200 mx-2" />
          <button 
            onClick={() => setShowLegend(!showLegend)}
            className={`p-2 rounded-lg ${showLegend ? 'bg-primary-100 text-primary-700' : 'hover:bg-gray-100 text-gray-500'}`}
          >
            {showLegend ? <Eye size={18} /> : <EyeOff size={18} />}
          </button>
          <button className="p-2 hover:bg-gray-100 rounded-lg text-gray-500">
            <Maximize2 size={18} />
          </button>
          <button className="p-2 hover:bg-gray-100 rounded-lg text-gray-500">
            <Download size={18} />
          </button>
        </div>
      </div>

      {/* Image Display */}
      <div className="flex gap-4">
        {/* Main Image Area */}
        <div className="flex-1">
          <div className="relative bg-gray-100 rounded-xl overflow-hidden" style={{ minHeight: '350px' }}>
            {activeTab === 'input' ? (
              originalImage ? (
                <img 
                  src={originalImage} 
                  alt="Input satellite image"
                  className="w-full h-full object-contain transition-transform duration-200"
                  style={{ transform: `scale(${zoom})` }}
                />
              ) : (
                <div className="flex flex-col items-center justify-center h-full min-h-[350px] text-gray-400">
                  <ImageIcon size={48} className="mb-4 opacity-50" />
                  <p className="font-medium">No image uploaded</p>
                  <p className="text-sm">Upload a satellite image to begin analysis</p>
                </div>
              )
            ) : (
              segmentedImage ? (
                <img 
                  src={segmentedImage} 
                  alt="Segmented crop map"
                  className="w-full h-full object-contain transition-transform duration-200"
                  style={{ transform: `scale(${zoom})` }}
                />
              ) : (
                <div className="flex flex-col items-center justify-center h-full min-h-[350px] text-gray-400">
                  <Layers size={48} className="mb-4 opacity-50" />
                  <p className="font-medium">No segmentation available</p>
                  <p className="text-sm">Upload and analyze an image first</p>
                </div>
              )
            )}
          </div>
        </div>

        {/* Legend Panel */}
        {showLegend && (
          <div className="w-64 bg-gray-50 rounded-xl p-4 animate-fadeIn">
            <h4 className="text-sm font-semibold text-gray-700 mb-3">
              Land Cover Classes
            </h4>
            <div className="space-y-2">
              {Object.entries(classColors).map(([className, data]) => {
                const isDetected = classesDetected?.includes(className);
                return (
                  <div 
                    key={className}
                    className={`
                      flex items-center gap-3 p-2 rounded-lg transition-all
                      ${isDetected ? 'bg-white shadow-sm' : 'opacity-50'}
                    `}
                  >
                    <div 
                      className="w-4 h-4 rounded-md shadow-sm flex-shrink-0"
                      style={{ backgroundColor: data.color }}
                    />
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium text-gray-700 truncate">
                        {className}
                      </p>
                      <p className="text-[10px] text-gray-400 truncate">
                        {data.description}
                      </p>
                    </div>
                    {isDetected && (
                      <div className="w-2 h-2 bg-primary-500 rounded-full flex-shrink-0" />
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Bottom Info Bar */}
      {(originalImage || segmentedImage) && (
        <div className="mt-4 flex items-center justify-between text-xs text-gray-500 px-2">
          <span>
            {classesDetected?.length || 0} classes detected
          </span>
          <span>
            Resolution: 256 × 256 px
          </span>
        </div>
      )}
    </div>
  );
};

export default SegmentationView;
