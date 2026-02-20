/**
 * ============================================================================
 * UPLOAD CARD COMPONENT
 * ============================================================================
 * Drag & drop image upload interface for satellite imagery
 */

import React, { useCallback, useState } from 'react';
import { 
  Upload, 
  Image as ImageIcon, 
  X, 
  CheckCircle,
  AlertCircle,
  Loader2
} from 'lucide-react';

const UploadCard = ({ onImageUpload, isLoading }) => {
  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState(null);
  const [fileName, setFileName] = useState('');
  const [error, setError] = useState('');

  // Handle file selection
  const handleFile = useCallback((file) => {
    setError('');
    
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/tiff', 'image/tif'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(png|jpg|jpeg|tif|tiff)$/i)) {
      setError('Please upload a valid image file (PNG, JPEG, or TIFF)');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB');
      return;
    }

    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(file);

    setFileName(file.name);
    onImageUpload(file);
  }, [onImageUpload]);

  // Drag and drop handlers
  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  }, [handleFile]);

  // File input change handler
  const handleInputChange = useCallback((e) => {
    const files = e.target.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  }, [handleFile]);

  // Clear selection
  const handleClear = useCallback(() => {
    setPreview(null);
    setFileName('');
    setError('');
  }, []);

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">
          Upload Satellite Image
        </h3>
        {preview && (
          <button 
            onClick={handleClear}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X size={20} />
          </button>
        )}
      </div>

      {/* Upload Zone */}
      {!preview ? (
        <div
          className={`upload-zone ${dragOver ? 'dragover' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-input').click()}
        >
          <input
            id="file-input"
            type="file"
            accept=".png,.jpg,.jpeg,.tif,.tiff"
            onChange={handleInputChange}
            className="hidden"
          />
          
          <div className="flex flex-col items-center">
            <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mb-4">
              <Upload size={28} className="text-primary-600" />
            </div>
            
            <p className="text-gray-700 font-medium mb-2">
              Drag & drop your satellite image here
            </p>
            <p className="text-sm text-gray-500 mb-4">
              or click to browse files
            </p>
            
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <ImageIcon size={14} />
              <span>Supports: PNG, JPEG, TIFF (max 10MB)</span>
            </div>
          </div>
        </div>
      ) : (
        /* Preview Section */
        <div className="space-y-4">
          <div className="relative rounded-xl overflow-hidden bg-gray-100">
            <img 
              src={preview} 
              alt="Preview" 
              className="w-full h-48 object-contain"
            />
            {isLoading && (
              <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                <div className="flex items-center gap-2 text-white">
                  <Loader2 size={24} className="animate-spin" />
                  <span>Processing...</span>
                </div>
              </div>
            )}
          </div>
          
          <div className="flex items-center justify-between p-3 bg-primary-50 rounded-lg">
            <div className="flex items-center gap-3">
              <CheckCircle size={20} className="text-primary-600" />
              <div>
                <p className="text-sm font-medium text-gray-800 truncate max-w-[200px]">
                  {fileName}
                </p>
                <p className="text-xs text-gray-500">Ready for analysis</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="mt-4 flex items-center gap-2 p-3 bg-red-50 rounded-lg text-red-700">
          <AlertCircle size={18} />
          <span className="text-sm">{error}</span>
        </div>
      )}

      {/* Tips Section */}
      <div className="mt-4 p-4 bg-gray-50 rounded-xl">
        <p className="text-xs font-semibold text-gray-600 mb-2">💡 Tips for best results:</p>
        <ul className="text-xs text-gray-500 space-y-1">
          <li>• Use Sentinel-2 satellite imagery</li>
          <li>• Higher resolution images yield better predictions</li>
          <li>• Cloud-free images recommended</li>
        </ul>
      </div>
    </div>
  );
};

export default UploadCard;
