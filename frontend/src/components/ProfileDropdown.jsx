/**
 * ============================================================================
 * PROFILE DROPDOWN COMPONENT
 * ============================================================================
 * Admin profile menu with Profile, Dashboard, and Logout options.
 * Closes on outside click.
 */

import React, { useEffect, useRef } from 'react';
import { User, LayoutDashboard, LogOut } from 'lucide-react';

const ProfileDropdown = ({ isOpen, onClose, onNavigate }) => {
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

  const handleLogout = () => {
    localStorage.clear();
    onClose();
    window.location.href = '/';
  };

  const menuItems = [
    {
      id: 'profile',
      label: 'Profile',
      icon: User,
      color: 'text-gray-600',
      onClick: () => {
        onClose();
      },
    },
    {
      id: 'dashboard',
      label: 'Dashboard',
      icon: LayoutDashboard,
      color: 'text-gray-600',
      onClick: () => {
        if (onNavigate) onNavigate('home');
        onClose();
      },
    },
    {
      id: 'logout',
      label: 'Logout',
      icon: LogOut,
      color: 'text-red-500',
      divider: true,
      onClick: handleLogout,
    },
  ];

  return (
    <div
      ref={dropdownRef}
      className="absolute right-0 top-full mt-2 w-56 bg-white dark:bg-gray-800 rounded-xl shadow-xl border border-gray-200 dark:border-gray-700 z-50 overflow-hidden animate-fadeIn"
    >
      {/* Profile Info */}
      <div className="px-4 py-3 border-b border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/80">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-primary-100 dark:bg-primary-900/40 rounded-full flex items-center justify-center">
            <User size={20} className="text-primary-700 dark:text-primary-400" />
          </div>
          <div>
            <p className="text-sm font-semibold text-gray-800 dark:text-white">Admin</p>
            <p className="text-xs text-gray-500 dark:text-gray-400">admin@cropanalytics.ai</p>
          </div>
        </div>
      </div>

      {/* Menu Items */}
      <div className="py-1">
        {menuItems.map((item) => {
          const Icon = item.icon;
          return (
            <React.Fragment key={item.id}>
              {item.divider && <div className="my-1 border-t border-gray-100 dark:border-gray-700" />}
              <button
                onClick={item.onClick}
                className="w-full flex items-center gap-3 px-4 py-2.5 text-sm hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
              >
                <Icon size={16} className={item.color} />
                <span className={`font-medium ${item.color}`}>{item.label}</span>
              </button>
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};

export default ProfileDropdown;
