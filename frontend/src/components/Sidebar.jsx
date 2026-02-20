/**
 * ============================================================================
 * SIDEBAR COMPONENT
 * ============================================================================
 * Navigation sidebar with menu items for different dashboard sections
 */

import React from 'react';
import { 
  Upload, 
  Map, 
  BarChart3, 
  LineChart, 
  Wheat,
  Home,
  FileText,
  HelpCircle,
  X,
  Sparkles
} from 'lucide-react';

const Sidebar = ({ activePage, setActivePage, isOpen, onClose }) => {
  const menuItems = [
    { id: 'home', label: 'Dashboard', icon: Home },
    { id: 'upload', label: 'Upload Image', icon: Upload },
    { id: 'cropmap', label: 'Crop Map', icon: Map },
    { id: 'ndvi', label: 'NDVI Analysis', icon: LineChart },
    { id: 'statistics', label: 'Crop Statistics', icon: BarChart3 },
    { id: 'yield', label: 'Yield Estimation', icon: Wheat },
    { id: 'llm-insights', label: 'LLM Insights', icon: Sparkles },
  ];

  const bottomItems = [
    { id: 'reports', label: 'Reports', icon: FileText },
    { id: 'help', label: 'Help & Support', icon: HelpCircle },
  ];

  const handleNavigation = (pageId) => {
    setActivePage(pageId);
    onClose();
  };

  return (
    <>
      {/* Overlay for mobile */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <aside className={`
        fixed lg:static inset-y-0 left-0 z-50
        w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700
        transform transition-all duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        flex flex-col h-screen
      `}>
        {/* Mobile Close Button */}
        <div className="lg:hidden flex justify-end p-4">
          <button onClick={onClose} className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg">
            <X size={24} className="text-gray-600 dark:text-gray-300" />
          </button>
        </div>

        {/* Logo Section (Mobile) */}
        <div className="lg:hidden px-6 pb-4 border-b border-gray-100 dark:border-gray-700">
          <h2 className="text-lg font-bold text-primary-700 dark:text-primary-400">Crop Analytics</h2>
        </div>

        {/* Main Navigation */}
        <nav className="flex-1 px-4 py-6 space-y-2 overflow-y-auto">
          <p className="px-4 text-xs font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wider mb-4">
            Main Menu
          </p>
          
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;
            
            return (
              <button
                key={item.id}
                onClick={() => handleNavigation(item.id)}
                className={`
                  w-full sidebar-item
                  ${isActive ? 'active' : ''}
                `}
              >
                <Icon size={20} />
                <span>{item.label}</span>
                {isActive && (
                  <div className="ml-auto w-2 h-2 bg-primary-500 rounded-full" />
                )}
              </button>
            );
          })}
        </nav>

        {/* Bottom Navigation */}
        <div className="px-4 py-6 border-t border-gray-100 dark:border-gray-700">
          <p className="px-4 text-xs font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wider mb-4">
            Support
          </p>
          
          {bottomItems.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;
            
            return (
              <button
                key={item.id}
                onClick={() => handleNavigation(item.id)}
                className={`w-full sidebar-item ${isActive ? 'active' : ''}`}
              >
                <Icon size={20} />
                <span>{item.label}</span>
                {isActive && (
                  <div className="ml-auto w-2 h-2 bg-primary-500 rounded-full" />
                )}
              </button>
            );
          })}
        </div>

        {/* Status Card */}
        <div className="px-4 pb-6">
          <div className="bg-gradient-to-br from-primary-500 to-primary-700 rounded-2xl p-4 text-white">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 bg-green-300 rounded-full animate-pulse" />
              <span className="text-sm font-medium">System Status</span>
            </div>
            <p className="text-xs text-primary-100">
              Model loaded & ready for predictions
            </p>
          </div>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
