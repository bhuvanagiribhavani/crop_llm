/**
 * ============================================================================
 * HEADER COMPONENT
 * ============================================================================
 * Top navigation bar with title, logo, and interactive action buttons.
 * Includes notification dropdown, settings modal, and profile menu.
 */

import React, { useState, useCallback } from 'react';
import { 
  Leaf, 
  Settings, 
  Bell, 
  User,
  Menu
} from 'lucide-react';
import NotificationDropdown from './NotificationDropdown';
import SettingsModal from './SettingsModal';
import ProfileDropdown from './ProfileDropdown';

const Header = ({ onMenuClick, onNavigate, notifications = [], onClearNotifications, onSettingsChange }) => {
  const [showNotifications, setShowNotifications] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showProfile, setShowProfile] = useState(false);

  // Close all dropdowns before opening one
  const closeAll = useCallback(() => {
    setShowNotifications(false);
    setShowProfile(false);
  }, []);

  const toggleNotifications = useCallback(() => {
    closeAll();
    setShowNotifications((prev) => !prev);
  }, [closeAll]);

  const toggleProfile = useCallback(() => {
    closeAll();
    setShowProfile((prev) => !prev);
  }, [closeAll]);

  const unreadCount = notifications.filter((n) => !n.read).length;

  return (
    <>
      <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4 flex items-center justify-between sticky top-0 z-40 transition-colors duration-300">
        {/* Left Section - Logo & Title */}
        <div className="flex items-center gap-4">
          {/* Mobile Menu Button */}
          <button 
            onClick={onMenuClick}
            className="lg:hidden p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
          >
            <Menu size={24} className="text-gray-600 dark:text-gray-300" />
          </button>

          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-700 rounded-xl flex items-center justify-center shadow-lg">
              <Leaf size={24} className="text-white" />
            </div>
            <div className="hidden sm:block">
              <h1 className="text-xl font-bold text-gray-800 dark:text-white">
                Crop Analytics
              </h1>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Sentinel-2 Satellite Imagery
              </p>
            </div>
          </div>
        </div>

        {/* Center Section - Title (Desktop) */}
        <div className="hidden md:block text-center">
          <h2 className="text-lg font-semibold text-gray-700 dark:text-gray-200">
            Agricultural Land Analysis Dashboard
          </h2>
        </div>

        {/* Right Section - Actions */}
        <div className="flex items-center gap-2">
          {/* Notifications */}
          <div className="relative">
            <button
              onClick={toggleNotifications}
              className={`p-2 rounded-lg relative transition-colors ${
                showNotifications ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400' : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300'
              }`}
            >
              <Bell size={20} />
              {unreadCount > 0 && (
                <span className="absolute top-0.5 right-0.5 min-w-[16px] h-4 bg-red-500 rounded-full flex items-center justify-center">
                  <span className="text-[10px] font-bold text-white leading-none">
                    {unreadCount > 9 ? '9+' : unreadCount}
                  </span>
                </span>
              )}
            </button>
            <NotificationDropdown
              isOpen={showNotifications}
              onClose={() => setShowNotifications(false)}
              notifications={notifications}
              onClear={onClearNotifications}
            />
          </div>

          {/* Settings */}
          <button
            onClick={() => setShowSettings(true)}
            className={`p-2 rounded-lg transition-colors ${
              showSettings ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400' : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300'
            }`}
          >
            <Settings size={20} />
          </button>

          {/* Profile */}
          <div className="relative">
            <button
              onClick={toggleProfile}
              className={`flex items-center gap-2 p-2 rounded-lg transition-colors ${
                showProfile ? 'bg-primary-50 dark:bg-primary-900/30' : 'hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              <div className="w-8 h-8 bg-primary-100 dark:bg-primary-900/40 rounded-full flex items-center justify-center">
                <User size={18} className="text-primary-700 dark:text-primary-400" />
              </div>
              <span className="hidden sm:block text-sm font-medium text-gray-700 dark:text-gray-200">
                Admin
              </span>
            </button>
            <ProfileDropdown
              isOpen={showProfile}
              onClose={() => setShowProfile(false)}
              onNavigate={onNavigate}
            />
          </div>
        </div>
      </header>

      {/* Settings Modal (rendered outside header for proper overlay) */}
      <SettingsModal
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        onSettingsChange={onSettingsChange}
      />
    </>
  );
};

export default Header;
