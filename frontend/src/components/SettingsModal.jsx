/**
 * ============================================================================
 * SETTINGS MODAL COMPONENT
 * ============================================================================
 * Modal with Dark Mode toggle, Notification toggle, and Save.
 * Persists settings to localStorage.
 */

import React, { useState, useEffect, useRef } from 'react';
import { X, Moon, Bell, Save, Check } from 'lucide-react';

const SETTINGS_KEY = 'crop_analytics_settings';

const defaultSettings = {
  darkMode: false,
  notificationsEnabled: true,
};

const loadSettings = () => {
  try {
    const stored = localStorage.getItem(SETTINGS_KEY);
    return stored ? { ...defaultSettings, ...JSON.parse(stored) } : defaultSettings;
  } catch {
    return defaultSettings;
  }
};

const SettingsModal = ({ isOpen, onClose, onSettingsChange }) => {
  const [settings, setSettings] = useState(loadSettings);
  const [saved, setSaved] = useState(false);
  const modalRef = useRef(null);

  // Load settings when modal opens
  useEffect(() => {
    if (isOpen) {
      setSettings(loadSettings());
      setSaved(false);
    }
  }, [isOpen]);

  // Close on outside click
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (modalRef.current && !modalRef.current.contains(e.target)) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, onClose]);

  // Close on Escape key
  useEffect(() => {
    const handleEsc = (e) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      document.addEventListener('keydown', handleEsc);
    }
    return () => document.removeEventListener('keydown', handleEsc);
  }, [isOpen, onClose]);

  const handleToggle = (key) => {
    setSettings((prev) => ({ ...prev, [key]: !prev[key] }));
    setSaved(false);
  };

  const handleSave = () => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    if (onSettingsChange) onSettingsChange(settings);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 animate-fadeIn">
      <div
        ref={modalRef}
        className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-white">Settings</h3>
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            <X size={18} className="text-gray-500 dark:text-gray-400" />
          </button>
        </div>

        {/* Settings Content */}
        <div className="px-6 py-5 space-y-5">
          {/* Dark Mode Toggle */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gray-100 dark:bg-gray-700 rounded-xl flex items-center justify-center">
                <Moon size={20} className="text-gray-600 dark:text-gray-300" />
              </div>
              <div>
                <p className="text-sm font-medium text-gray-800 dark:text-gray-100">Dark Mode</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">Switch to dark theme</p>
              </div>
            </div>
            <button
              onClick={() => handleToggle('darkMode')}
              className={`relative w-12 h-6 rounded-full transition-colors duration-300 ${
                settings.darkMode ? 'bg-primary-500' : 'bg-gray-300'
              }`}
            >
              <span
                className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform duration-300 ${
                  settings.darkMode ? 'translate-x-6' : 'translate-x-0'
                }`}
              />
            </button>
          </div>

          {/* Notifications Toggle */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-50 dark:bg-blue-900/30 rounded-xl flex items-center justify-center">
                <Bell size={20} className="text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <p className="text-sm font-medium text-gray-800 dark:text-gray-100">Notifications</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">Show activity alerts</p>
              </div>
            </div>
            <button
              onClick={() => handleToggle('notificationsEnabled')}
              className={`relative w-12 h-6 rounded-full transition-colors duration-300 ${
                settings.notificationsEnabled ? 'bg-primary-500' : 'bg-gray-300'
              }`}
            >
              <span
                className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform duration-300 ${
                  settings.notificationsEnabled ? 'translate-x-6' : 'translate-x-0'
                }`}
              />
            </button>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/80 flex items-center justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors">
            Cancel
          </button>
          <button
            onClick={handleSave}
            className={`px-5 py-2 text-sm font-medium rounded-lg flex items-center gap-2 transition-all duration-300 ${
              saved
                ? 'bg-green-500 text-white'
                : 'bg-primary-600 text-white hover:bg-primary-700'
            }`}
          >
            {saved ? (
              <>
                <Check size={16} />
                Saved
              </>
            ) : (
              <>
                <Save size={16} />
                Save
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
