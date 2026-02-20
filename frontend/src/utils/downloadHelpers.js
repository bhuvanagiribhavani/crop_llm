/**
 * ============================================================================
 * DOWNLOAD HELPER UTILITIES
 * ============================================================================
 * Utility functions for generating and downloading reports in various formats
 * 
 * Supported formats:
 * - PDF (using jsPDF library)
 * - CSV (comma-separated values)
 * - JSON (JavaScript Object Notation)
 * - Excel (using XML spreadsheet format)
 */

import { jsPDF } from 'jspdf';

/**
 * Generate and download a Full Analysis PDF report
 * @param {string} filename - Name of the file to download
 * @param {object} data - Report data to include
 * @returns {boolean} Success status
 */
export const downloadFullAnalysisPDF = (filename, data) => {
  try {
    // Create new PDF document
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    let yPos = 20;

    // Title
    doc.setFontSize(22);
    doc.setTextColor(34, 139, 34); // Forest green
    doc.text('CROP ANALYTICS', pageWidth / 2, yPos, { align: 'center' });
    yPos += 10;
    
    doc.setFontSize(16);
    doc.setTextColor(0, 0, 0);
    doc.text('Full Analysis Report', pageWidth / 2, yPos, { align: 'center' });
    yPos += 15;

    // Date and metadata
    doc.setFontSize(10);
    doc.setTextColor(100, 100, 100);
    doc.text(`Generated: ${new Date().toLocaleDateString()} at ${new Date().toLocaleTimeString()}`, pageWidth / 2, yPos, { align: 'center' });
    yPos += 5;
    doc.text('System: Crop Analytics Dashboard v1.0 | Model: U-Net Segmentation', pageWidth / 2, yPos, { align: 'center' });
    yPos += 15;

    // Horizontal line
    doc.setDrawColor(34, 139, 34);
    doc.setLineWidth(0.5);
    doc.line(20, yPos, pageWidth - 20, yPos);
    yPos += 15;

    // Land Cover Statistics Section
    doc.setFontSize(14);
    doc.setTextColor(0, 0, 0);
    doc.text('LAND COVER STATISTICS', 20, yPos);
    yPos += 10;

    doc.setFontSize(10);
    if (data.statistics) {
      Object.entries(data.statistics).forEach(([className, stats]) => {
        if (yPos > 270) {
          doc.addPage();
          yPos = 20;
        }
        doc.setTextColor(60, 60, 60);
        doc.text(`• ${className}`, 25, yPos);
        doc.text(`Coverage: ${stats.percentage?.toFixed(2) || 0}%`, 80, yPos);
        doc.text(`Pixels: ${stats.pixel_count?.toLocaleString() || 0}`, 140, yPos);
        yPos += 7;
      });
    } else {
      doc.text('No statistics data available', 25, yPos);
      yPos += 7;
    }
    yPos += 10;

    // AI Insights Section
    doc.setFontSize(14);
    doc.setTextColor(0, 0, 0);
    doc.text('AI-POWERED INSIGHTS', 20, yPos);
    yPos += 10;

    doc.setFontSize(10);
    doc.setTextColor(60, 60, 60);
    if (data.insight) {
      const insightLines = doc.splitTextToSize(data.insight, pageWidth - 40);
      insightLines.forEach((line) => {
        if (yPos > 270) {
          doc.addPage();
          yPos = 20;
        }
        doc.text(line, 20, yPos);
        yPos += 6;
      });
    } else {
      doc.text('No AI insights generated yet. Upload an image for analysis.', 20, yPos);
      yPos += 6;
    }
    yPos += 10;

    // Detected Classes
    if (data.classesDetected && data.classesDetected.length > 0) {
      doc.setFontSize(14);
      doc.setTextColor(0, 0, 0);
      doc.text('DETECTED CLASSES', 20, yPos);
      yPos += 10;
      
      doc.setFontSize(10);
      doc.setTextColor(60, 60, 60);
      doc.text(data.classesDetected.join(', '), 20, yPos);
      yPos += 15;
    }

    // Footer
    const pageCount = doc.internal.getNumberOfPages();
    for (let i = 1; i <= pageCount; i++) {
      doc.setPage(i);
      doc.setFontSize(8);
      doc.setTextColor(150, 150, 150);
      doc.text(
        `© 2026 Crop Analytics Project | Page ${i} of ${pageCount}`,
        pageWidth / 2,
        290,
        { align: 'center' }
      );
    }

    // Save the PDF
    doc.save(filename);
    return true;
  } catch (error) {
    console.error('PDF download error:', error);
    return false;
  }
};

/**
 * Generate and download a Crop Health Summary PDF report
 * @param {string} filename - Name of the file to download
 * @param {object} data - Report data to include
 * @returns {boolean} Success status
 */
export const downloadCropHealthSummaryPDF = (filename, data) => {
  try {
    // Create new PDF document
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    let yPos = 20;

    // Header with green theme
    doc.setFillColor(34, 139, 34);
    doc.rect(0, 0, pageWidth, 40, 'F');
    
    doc.setFontSize(24);
    doc.setTextColor(255, 255, 255);
    doc.text('Crop Health Summary', pageWidth / 2, 25, { align: 'center' });
    
    doc.setFontSize(10);
    doc.text(`Report Date: ${new Date().toLocaleDateString()}`, pageWidth / 2, 35, { align: 'center' });
    
    yPos = 55;

    // Health Score Box
    const cropPercentage = data.statistics?.['Crops']?.percentage || 35.8;
    const healthScore = Math.min(100, Math.round(50 + cropPercentage));
    
    doc.setFillColor(240, 255, 240);
    doc.roundedRect(20, yPos, pageWidth - 40, 35, 5, 5, 'F');
    
    doc.setFontSize(14);
    doc.setTextColor(34, 139, 34);
    doc.text('CROP HEALTH INDEX', pageWidth / 2, yPos + 12, { align: 'center' });
    
    doc.setFontSize(28);
    doc.setTextColor(0, 128, 0);
    doc.text(`${healthScore}/100`, pageWidth / 2, yPos + 28, { align: 'center' });
    
    yPos += 50;

    // Key Metrics Section
    doc.setFontSize(14);
    doc.setTextColor(0, 0, 0);
    doc.text('KEY AGRICULTURAL METRICS', 20, yPos);
    yPos += 12;

    const metrics = [
      { label: 'Total Crop Coverage', value: `${cropPercentage.toFixed(1)}%` },
      { label: 'Vegetation Health', value: cropPercentage > 30 ? 'Good' : 'Moderate' },
      { label: 'Water Availability', value: `${data.statistics?.['Water']?.percentage?.toFixed(1) || 8.5}%` },
      { label: 'Tree Coverage', value: `${data.statistics?.['Trees']?.percentage?.toFixed(1) || 22.3}%` },
      { label: 'Built Area Impact', value: `${data.statistics?.['Built Area']?.percentage?.toFixed(1) || 5.9}%` },
    ];

    doc.setFontSize(11);
    metrics.forEach((metric) => {
      doc.setTextColor(80, 80, 80);
      doc.text(`• ${metric.label}:`, 25, yPos);
      doc.setTextColor(34, 139, 34);
      doc.text(metric.value, 120, yPos);
      yPos += 9;
    });
    yPos += 10;

    // NDVI Analysis Section
    doc.setFontSize(14);
    doc.setTextColor(0, 0, 0);
    doc.text('NDVI VEGETATION ANALYSIS', 20, yPos);
    yPos += 12;

    doc.setFontSize(10);
    doc.setTextColor(60, 60, 60);
    const ndviText = `Based on vegetation coverage analysis, the estimated NDVI value is ${(cropPercentage / 100 * 0.7 + 0.2).toFixed(2)}. ` +
      `This indicates ${cropPercentage > 30 ? 'healthy and dense' : 'moderate'} vegetation in the analyzed region. ` +
      `The crop areas show ${cropPercentage > 40 ? 'excellent' : 'good'} photosynthetic activity.`;
    
    const ndviLines = doc.splitTextToSize(ndviText, pageWidth - 40);
    ndviLines.forEach((line) => {
      doc.text(line, 20, yPos);
      yPos += 6;
    });
    yPos += 15;

    // Recommendations
    doc.setFontSize(14);
    doc.setTextColor(0, 0, 0);
    doc.text('RECOMMENDATIONS', 20, yPos);
    yPos += 12;

    doc.setFontSize(10);
    const recommendations = [
      '1. Maintain current irrigation schedule for optimal crop growth',
      '2. Monitor boundary areas between crops and natural vegetation',
      '3. Consider soil nutrient analysis for yield optimization',
      '4. Schedule pest surveillance in high-density crop zones',
    ];

    doc.setTextColor(60, 60, 60);
    recommendations.forEach((rec) => {
      doc.text(rec, 20, yPos);
      yPos += 8;
    });

    // Footer
    doc.setFontSize(8);
    doc.setTextColor(150, 150, 150);
    doc.text(
      '© 2026 Crop Analytics Dashboard | Sentinel-2 Satellite Imagery Analysis',
      pageWidth / 2,
      285,
      { align: 'center' }
    );
    doc.text(
      'This report is generated using U-Net deep learning segmentation model',
      pageWidth / 2,
      290,
      { align: 'center' }
    );

    // Save the PDF
    doc.save(filename);
    return true;
  } catch (error) {
    console.error('Crop Health PDF error:', error);
    return false;
  }
};

/**
 * Legacy wrapper for backward compatibility
 */
export const downloadPDF = (filename, title, data) => {
  if (title.toLowerCase().includes('health')) {
    return downloadCropHealthSummaryPDF(filename, data);
  }
  return downloadFullAnalysisPDF(filename, data);
};

/**
 * Generate and download a CSV file
 * @param {string} filename - Name of the file to download
 * @param {object} data - Data to convert to CSV
 * @returns {boolean} Success status
 */
export const downloadCSV = (filename, data) => {
  try {
    let csvContent = '';
    
    if (data.statistics) {
      // Header row
      csvContent = 'Class Name,Coverage Percentage,Pixel Count,Hex Color\n';
      
      // Data rows
      Object.entries(data.statistics).forEach(([className, stats]) => {
        csvContent += `"${className}",${stats.percentage?.toFixed(4) || 0},${stats.pixel_count || 0},"${stats.color || '#000000'}"\n`;
      });
    } else if (data.ndviTimeSeries) {
      // NDVI time series format
      csvContent = 'Month,NDVI Value,Rainfall (mm),Temperature (°C)\n';
      data.ndviTimeSeries.forEach(row => {
        csvContent += `${row.month},${row.ndvi},${row.rainfall},${row.temperature || 25}\n`;
      });
    } else {
      // Generic data export
      csvContent = 'Key,Value\n';
      Object.entries(data).forEach(([key, value]) => {
        csvContent += `"${key}","${JSON.stringify(value)}"\n`;
      });
    }
    
    // Create blob and download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    triggerDownload(blob, filename);
    
    return true;
  } catch (error) {
    console.error('CSV download error:', error);
    return false;
  }
};

/**
 * Generate and download a JSON file
 * @param {string} filename - Name of the file to download
 * @param {object} data - Data to export as JSON
 * @returns {boolean} Success status
 */
export const downloadJSON = (filename, data) => {
  try {
    // Format JSON with indentation for readability
    const jsonContent = JSON.stringify({
      metadata: {
        generated: new Date().toISOString(),
        source: 'Crop Analytics Dashboard',
        version: '1.0.0'
      },
      data: data
    }, null, 2);
    
    // Create blob and download
    const blob = new Blob([jsonContent], { type: 'application/json' });
    triggerDownload(blob, filename);
    
    return true;
  } catch (error) {
    console.error('JSON download error:', error);
    return false;
  }
};

/**
 * Generate and download an Excel file (simplified XLSX format)
 * Uses XML-based spreadsheet format compatible with Excel
 * @param {string} filename - Name of the file to download
 * @param {object} data - Data to export to Excel
 * @returns {boolean} Success status
 */
export const downloadExcel = (filename, data) => {
  try {
    // Create Excel-compatible XML content
    let excelContent = generateExcelXML(data);
    
    // Create blob with Excel mime type
    const blob = new Blob([excelContent], { 
      type: 'application/vnd.ms-excel' 
    });
    triggerDownload(blob, filename);
    
    return true;
  } catch (error) {
    console.error('Excel download error:', error);
    return false;
  }
};

/**
 * Generate Excel XML content
 * @param {object} data - Data to convert to Excel format
 * @returns {string} Excel XML content
 */
const generateExcelXML = (data) => {
  let xml = `<?xml version="1.0" encoding="UTF-8"?>
<?mso-application progid="Excel.Sheet"?>
<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"
 xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">
 <Worksheet ss:Name="NDVI Time Series">
  <Table>
   <Row>
    <Cell><Data ss:Type="String">Month</Data></Cell>
    <Cell><Data ss:Type="String">NDVI Value</Data></Cell>
    <Cell><Data ss:Type="String">Rainfall (mm)</Data></Cell>
    <Cell><Data ss:Type="String">Status</Data></Cell>
   </Row>`;

  if (data.ndviTimeSeries) {
    data.ndviTimeSeries.forEach(row => {
      xml += `
   <Row>
    <Cell><Data ss:Type="String">${row.month}</Data></Cell>
    <Cell><Data ss:Type="Number">${row.ndvi}</Data></Cell>
    <Cell><Data ss:Type="Number">${row.rainfall}</Data></Cell>
    <Cell><Data ss:Type="String">${row.status || 'Normal'}</Data></Cell>
   </Row>`;
    });
  }

  xml += `
  </Table>
 </Worksheet>
</Workbook>`;

  return xml;
};

/**
 * Trigger file download in browser
 * @param {Blob} blob - File blob to download
 * @param {string} filename - Name for the downloaded file
 */
const triggerDownload = (blob, filename) => {
  // Create temporary URL for the blob
  const url = URL.createObjectURL(blob);
  
  // Create temporary link element
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  
  // Append to body, click, and remove
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  
  // Clean up the URL object
  setTimeout(() => URL.revokeObjectURL(url), 100);
};

/**
 * Generate mock NDVI time series data
 * @returns {Array} Array of monthly NDVI data
 */
export const generateMockNDVIData = () => {
  return [
    { month: 'January', ndvi: 0.35, rainfall: 45, status: 'Low' },
    { month: 'February', ndvi: 0.38, rainfall: 52, status: 'Low' },
    { month: 'March', ndvi: 0.45, rainfall: 78, status: 'Moderate' },
    { month: 'April', ndvi: 0.52, rainfall: 95, status: 'Moderate' },
    { month: 'May', ndvi: 0.68, rainfall: 110, status: 'Good' },
    { month: 'June', ndvi: 0.75, rainfall: 85, status: 'Excellent' },
    { month: 'July', ndvi: 0.72, rainfall: 60, status: 'Excellent' },
    { month: 'August', ndvi: 0.65, rainfall: 45, status: 'Good' },
    { month: 'September', ndvi: 0.58, rainfall: 55, status: 'Good' },
    { month: 'October', ndvi: 0.48, rainfall: 70, status: 'Moderate' },
    { month: 'November', ndvi: 0.42, rainfall: 65, status: 'Moderate' },
    { month: 'December', ndvi: 0.36, rainfall: 50, status: 'Low' },
  ];
};

/**
 * Generate mock statistics data for demo
 * @returns {object} Mock statistics object
 */
export const generateMockStatistics = () => {
  return {
    'Water': { percentage: 8.5, pixel_count: 5500, color: '#419bdf' },
    'Trees': { percentage: 22.3, pixel_count: 14500, color: '#397d49' },
    'Grass': { percentage: 12.7, pixel_count: 8250, color: '#88b053' },
    'Flooded Vegetation': { percentage: 3.2, pixel_count: 2080, color: '#7a87c6' },
    'Crops': { percentage: 35.8, pixel_count: 23270, color: '#e49635' },
    'Scrub/Shrub': { percentage: 8.1, pixel_count: 5265, color: '#dfc35a' },
    'Built Area': { percentage: 5.9, pixel_count: 3835, color: '#c4281b' },
    'Bare Ground': { percentage: 3.5, pixel_count: 2275, color: '#a59b8f' },
  };
};

/**
 * Generate mock insight text
 * @returns {string} AI-generated insight text
 */
export const generateMockInsight = () => {
  return `🌾 CROP HEALTH ANALYSIS SUMMARY

Based on the satellite imagery analysis, the agricultural area shows healthy vegetation patterns with 35.8% crop coverage.

KEY FINDINGS:
• Dominant land cover: Crops (35.8%) and Trees (22.3%)
• Water availability: Adequate (8.5% water coverage detected)
• Vegetation health: Good - Dense canopy observed in agricultural zones
• Risk areas: Minimal bare ground exposure (3.5%)

RECOMMENDATIONS:
1. Continue current irrigation schedule - water levels are optimal
2. Monitor flooded vegetation areas for potential drainage issues
3. Consider crop rotation in scrub/shrub boundary areas
4. Schedule fertilizer application during next growth phase

CROP HEALTH INDEX: 78/100 (Good)

This analysis is based on Sentinel-2 RGB imagery processed through our U-Net segmentation model.`;
};

// Named export object for convenience
const downloadHelpers = {
  downloadPDF,
  downloadCSV,
  downloadJSON,
  downloadExcel,
  generateMockNDVIData,
  generateMockStatistics,
  generateMockInsight
};

export default downloadHelpers;
