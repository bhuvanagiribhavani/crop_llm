/**
 * ============================================================================
 * NOTIFICATION DROPDOWN COMPONENT
 * ============================================================================
 * Displays recent prediction/analysis notifications in a dropdown panel.
 * Closes when clicking outside.
 */

import React, { useEffect, useRef } from 'react';
import { Bell, X, Leaf, BarChart3, Image, Trash2 } from 'lucide-react';

const ICON_MAP = {
  prediction: { icon: Leaf, color: 'bg-green-100 text-green-600' },
  analysis: { icon: BarChart3, color: 'bg-blue-100 text-blue-600' },
  upload: { icon: Image, color: 'bg-orange-100 text-orange-600' },
};

const NotificationDropdown = ({ isOpen, onClose, notifications, onClear }) => {
  const dropdownRef = useRef(null);

  // Close on outside click
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      ref={dropdownRef}
      className="absolute right-0 top-full mt-2 w-80 bg-white dark:bg-gray-800 rounded-xl shadow-xl border border-gray-200 dark:border-gray-700 z-50 overflow-hidden animate-fadeIn"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/80">
        <h3 className="text-sm font-semibold text-gray-800 dark:text-white">Notifications</h3>
        <div className="flex items-center gap-2">
          {notifications.length > 0 && (
            <button
              onClick={onClear}
              className="text-xs text-gray-500 hover:text-red-500 flex items-center gap-1 transition-colors"
              title="Clear all"
            >
              <Trash2 size={12} />
              Clear
            </button>
          )}
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-lg transition-colors"
          >
            <X size={14} className="text-gray-500 dark:text-gray-400" />
          </button>
        </div>
      </div>

      {/* Notification List */}
      <div className="max-h-72 overflow-y-auto">
        {notifications.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-gray-400">
            <Bell size={32} className="mb-2 opacity-40" />
            <p className="text-sm font-medium">No new notifications</p>
            <p className="text-xs mt-1">Activity will appear here</p>
          </div>
        ) : (
          notifications.map((notif) => {
            const { icon: Icon, color } = ICON_MAP[notif.type] || ICON_MAP.prediction;
            return (
              <div
                key={notif.id}
                className={`flex items-start gap-3 px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors border-b border-gray-50 dark:border-gray-700/50 ${
                  !notif.read ? 'bg-primary-50/30 dark:bg-primary-900/20' : ''
                }`}
              >
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${color}`}>
                  <Icon size={16} />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-800 dark:text-gray-100 font-medium truncate">
                    {notif.title}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">{notif.message}</p>
                  <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">{notif.time}</p>
                </div>
                {!notif.read && (
                  <span className="w-2 h-2 bg-primary-500 rounded-full mt-1.5 flex-shrink-0"></span>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default NotificationDropdown;
