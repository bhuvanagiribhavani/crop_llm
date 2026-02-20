/**
 * ============================================================================
 * STATISTICS PANEL COMPONENT
 * ============================================================================
 * Display crop statistics with progress bars and charts
 */

import React from 'react';
import { 
  PieChart, 
  Pie, 
  Cell, 
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend
} from 'recharts';
import { 
  TrendingUp, 
  Droplets, 
  Trees, 
  Wheat,
  Building,
  Mountain
} from 'lucide-react';

const StatisticsPanel = ({ statistics }) => {
  // Default statistics if none provided
  const defaultStats = {
    'Water': { percentage: 0, pixel_count: 0, area_hectares: 0 },
    'Trees': { percentage: 0, pixel_count: 0, area_hectares: 0 },
    'Grass': { percentage: 0, pixel_count: 0, area_hectares: 0 },
    'Flooded Vegetation': { percentage: 0, pixel_count: 0, area_hectares: 0 },
    'Crops': { percentage: 0, pixel_count: 0, area_hectares: 0 },
    'Scrub/Shrub': { percentage: 0, pixel_count: 0, area_hectares: 0 },
    'Built Area': { percentage: 0, pixel_count: 0, area_hectares: 0 },
    'Bare Ground': { percentage: 0, pixel_count: 0, area_hectares: 0 },
  };

  const stats = statistics || defaultStats;

  // Color mapping
  const colors = {
    'Water': '#419BDF',
    'Trees': '#397D49',
    'Grass': '#88B053',
    'Flooded Vegetation': '#7A87C6',
    'Crops': '#E4963A',
    'Scrub/Shrub': '#DFC17D',
    'Built Area': '#C4281B',
    'Bare Ground': '#A59B8F',
  };

  // Icon mapping
  const icons = {
    'Water': Droplets,
    'Trees': Trees,
    'Grass': Mountain,
    'Crops': Wheat,
    'Built Area': Building,
  };

  // Prepare data for charts
  const pieData = Object.entries(stats)
    .filter(([_, data]) => data.percentage > 0)
    .map(([name, data]) => ({
      name,
      value: data.percentage,
      color: colors[name]
    }));

  const barData = Object.entries(stats)
    .filter(([_, data]) => data.percentage > 0)
    .map(([name, data]) => ({
      name: name.length > 10 ? name.substring(0, 10) + '...' : name,
      fullName: name,
      area: data.area_hectares,
      color: colors[name]
    }));

  // Calculate totals
  const totalArea = Object.values(stats).reduce((sum, s) => sum + (s.area_hectares || 0), 0);
  const vegetationPct = (stats['Trees']?.percentage || 0) + 
                        (stats['Grass']?.percentage || 0) + 
                        (stats['Crops']?.percentage || 0) +
                        (stats['Flooded Vegetation']?.percentage || 0) +
                        (stats['Scrub/Shrub']?.percentage || 0);

  // Key metrics
  const keyMetrics = [
    { 
      label: 'Total Area', 
      value: `${totalArea.toFixed(1)} ha`, 
      icon: TrendingUp,
      color: 'bg-blue-100 text-blue-700'
    },
    { 
      label: 'Crops', 
      value: `${(stats['Crops']?.percentage || 0).toFixed(1)}%`, 
      icon: Wheat,
      color: 'bg-orange-100 text-orange-700'
    },
    { 
      label: 'Vegetation', 
      value: `${vegetationPct.toFixed(1)}%`, 
      icon: Trees,
      color: 'bg-green-100 text-green-700'
    },
    { 
      label: 'Water Bodies', 
      value: `${(stats['Water']?.percentage || 0).toFixed(1)}%`, 
      icon: Droplets,
      color: 'bg-cyan-100 text-cyan-700'
    },
  ];

  return (
    <div className="space-y-4">
      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {keyMetrics.map((metric, idx) => {
          const Icon = metric.icon;
          return (
            <div key={idx} className="stat-card">
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${metric.color}`}>
                  <Icon size={20} />
                </div>
                <div>
                  <p className="text-xs text-gray-500">{metric.label}</p>
                  <p className="text-lg font-bold text-gray-800">{metric.value}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Pie Chart */}
        <div className="card">
          <h4 className="text-sm font-semibold text-gray-700 mb-4">
            Land Cover Distribution
          </h4>
          <div className="h-64">
            {pieData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    formatter={(value) => [`${value.toFixed(1)}%`, 'Coverage']}
                    contentStyle={{
                      borderRadius: '12px',
                      border: 'none',
                      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                    }}
                  />
                  <Legend 
                    verticalAlign="bottom" 
                    height={36}
                    formatter={(value) => <span className="text-xs text-gray-600">{value}</span>}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-gray-400">
                <p className="text-sm">No data available</p>
              </div>
            )}
          </div>
        </div>

        {/* Bar Chart */}
        <div className="card">
          <h4 className="text-sm font-semibold text-gray-700 mb-4">
            Area by Class (Hectares)
          </h4>
          <div className="h-64">
            {barData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={barData} layout="vertical">
                  <XAxis type="number" tick={{ fontSize: 10 }} />
                  <YAxis 
                    type="category" 
                    dataKey="name" 
                    tick={{ fontSize: 10 }} 
                    width={80}
                  />
                  <Tooltip 
                    formatter={(value, name, props) => [
                      `${value.toFixed(2)} ha`, 
                      props.payload.fullName
                    ]}
                    contentStyle={{
                      borderRadius: '12px',
                      border: 'none',
                      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                    }}
                  />
                  <Bar 
                    dataKey="area" 
                    radius={[0, 4, 4, 0]}
                  >
                    {barData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-gray-400">
                <p className="text-sm">No data available</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Detailed Breakdown */}
      <div className="card">
        <h4 className="text-sm font-semibold text-gray-700 mb-4">
          Detailed Breakdown
        </h4>
        <div className="space-y-3">
          {Object.entries(stats)
            .filter(([_, data]) => data.percentage > 0)
            .sort((a, b) => b[1].percentage - a[1].percentage)
            .map(([name, data]) => (
              <div key={name} className="flex items-center gap-4">
                <div 
                  className="w-3 h-3 rounded-full flex-shrink-0"
                  style={{ backgroundColor: colors[name] }}
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-gray-700">{name}</span>
                    <span className="text-sm text-gray-500">
                      {data.percentage.toFixed(1)}% ({data.area_hectares.toFixed(2)} ha)
                    </span>
                  </div>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill"
                      style={{ 
                        width: `${data.percentage}%`,
                        backgroundColor: colors[name]
                      }}
                    />
                  </div>
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
};

export default StatisticsPanel;
