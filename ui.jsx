import React, { useState, useEffect, useRef } from 'react';
import { Chart, RadarController, LineElement, PointElement, LinearScale, CategoryScale, Title, Tooltip, Legend,  RadialLinearScale,  } from 'chart.js';
import { ArrowRight, CheckCircle, AlertTriangle, Clock, Calendar, Lightbulb } from 'lucide-react';

// Register Chart.js components
Chart.register(RadarController, RadialLinearScale, LineElement, PointElement, LinearScale, CategoryScale, Title, Tooltip, Legend);

const UiUxAnalysisDashboard = ({ resultData }) => {
  const [activeTab, setActiveTab] = useState('summary');
  const radarChartRef = useRef(null);
  const chartInstanceRef = useRef(null);

   if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
    }

  useEffect(() => {
    if (!resultData) return;

    // Calculate stroke-dashoffset for the progress rings
    const calculateScoreOffset = (score) => {
      const circumference = 2 * Math.PI * 34;
      return circumference - (score / 10) * circumference;
    };

    // Update the progress rings
    const overallScoreCircle = document.getElementById('overall-score-circle');
    if (overallScoreCircle) {
      overallScoreCircle.style.strokeDashoffset = calculateScoreOffset(resultData.result?.overall_score);
    }

    // Initialize radar chart
    if (radarChartRef.current && resultData) {
      const scores = {
        layout: resultData.result?.detailed_results?.layout?.overall_score,
        color: resultData.result?.detailed_results?.color?.overall_score,
        typography: resultData.result?.detailed_results?.typography?.overall_score,
        accessibility: resultData.result?.detailed_results?.accessibility?.overall_score,
        usability: resultData.result?.detailed_results?.usability?.overall_score,
      };


      const ctx = radarChartRef.current.getContext('2d');


      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
      }

      chartInstanceRef.current = new Chart(ctx, {
        type: 'radar',
        data: {
          labels: ['Layout', 'Color', 'Typography', 'Accessibility', 'Usability'],
          datasets: [{
            label: 'Score',
            data: [scores?.layout, scores?.color, scores?.typography, scores?.accessibility, scores?.usability],
            backgroundColor: 'rgba(51, 102, 204, 0.2)',
            borderColor: 'rgba(51, 102, 204, 1)',
            pointBackgroundColor: 'rgba(51, 102, 204, 1)',
          }]
        },
        options: {
          scales: {
            r: {
              angleLines: { color: 'rgba(0, 0, 0, 0.1)' },
              grid: { color: 'rgba(0, 0, 0, 0.1)' },
              pointLabels: { font: { size: 12 } },
              suggestedMin: 0,
              suggestedMax: 10
            }
          },
          plugins: {
            legend: { display: false },
            tooltip: { enabled: true }
          }
        }
      });
    }

    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
      }
    };
  }, [resultData, activeTab]);

  if (!resultData) return <div className="p-6">Loading analysis data...</div>;

  const getScoreIndicator = (score) => {
    if (score >= 7) return { text: 'Good', icon: 'check-circle', color: 'text-green-500' };
    if (score >= 5) return { text: 'Needs Work', icon: 'exclamation-circle', color: 'text-yellow-500' };
    return { text: 'Critical', icon: 'exclamation-triangle', color: 'text-red-500' };
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getPriorityItems = (priority) => {
    return resultData.result?.improvement_priority
      .filter(item => item.priority === priority)
      .map((item, index) => (
        <div key={`${item.module}-${index}`} className="improvement-card bg-white border-l-4 border-red-500 rounded-lg shadow-sm p-4 transition-all duration-200">
          <div className="flex justify-between items-start">
            <div>
              <div className="text-gray-500 text-sm mb-1">{item.module.charAt(0).toUpperCase() + item.module.slice(1)}</div>
              <h3 className="font-semibold text-lg">{item.issue}</h3>
            </div>
            <div className="flex items-center">
              <span className={`bg-red-100 text-red-800 text-xs font-medium mr-2 px-2.5 py-0.5 rounded-full`}>
                Priority {item.priority}
              </span>
            </div>
          </div>
          <p className="text-gray-600 mt-2">{item.recommendation}</p>
          <div className="mt-3 pt-3 border-t border-gray-100 flex justify-end">
            <button className="text-blue-600 hover:text-blue-800 text-sm font-medium flex items-center">
              View details <ArrowRight className="ml-1 h-4 w-4" />
            </button>
          </div>
        </div>
      ));
  };

  const getSummaryStrengths = () => {
    const strengths = [];
    
    // Layout strengths
    const layoutStrengths = resultData.result?.detailed_results?.layout.strengths || [];
    layoutStrengths.forEach(strength => {
      strengths.push({
        title: strength.description?.replace('score', '') || 'Visual Hierarchy',
        description: strength.explanation || 'Clear structure guides users through the dashboard effectively.'
      });
    });
    
    // Color strengths
    strengths.push({
      title: 'Color Readability',
      description: 'White background (#FFFFFF) ensures high visibility of task elements.'
    });
    
    strengths.push({
      title: 'Neutral Accents',
      description: '#F5F5F5 adds visual interest without overwhelming the user.'
    });
    
    return strengths.slice(0, 4); // Only return the first 4 strengths
  };

  // Get color related to priority
  const getPriorityColor = (priority) => {
    switch(priority) {
      case 5: return 'border-red-500';
      case 4: return 'border-amber-500';
      case 3: return 'border-blue-500';
      default: return 'border-gray-500';
    }
  };

  const getPriorityBg = (priority) => {
    switch(priority) {
      case 5: return 'bg-red-500';
      case 4: return 'bg-amber-500';
      case 3: return 'bg-blue-500';
      default: return 'bg-gray-500';
    }
  };

  const getPriorityBadge = (priority) => {
    switch(priority) {
      case 5: return 'bg-red-100 text-red-800';
      case 4: return 'bg-amber-100 text-amber-800';
      case 3: return 'bg-blue-100 text-blue-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="bg-gray-50 text-gray-800 font-sans w-full">
      <div className="min-h-screen flex flex-col">
        {/* Navigation Header */}
        <header className="bg-gray-800 text-white shadow-md">
          <div className="container mx-auto px-4 py-3 flex justify-between items-center">
            <div className="flex items-center space-x-4">
              <div className="text-2xl font-bold">UI/UX Analysis</div>
              <div className="text-gray-400 text-sm">Dashboard <span className="px-2">•</span> {resultData?.result?.design_context?.estimated_purpose}</div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-grow container mx-auto px-4 py-6">
          {/* Dashboard Header */}
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8">
            <div>
              <h1 className="text-3xl font-bold">{resultData?.result?.design_context?.estimated_purpose} Analysis</h1>
              <p className="text-gray-600 mt-2">Last updated: {formatDate(resultData.result?.timestamp)}</p>
            </div>
            <div className="mt-4 md:mt-0 flex items-center space-x-4">
              <div className="flex items-center">
                <div className="mr-4 relative">
                  <svg className="progress-ring" width="80" height="80">
                    <circle className="progress-ring__circle-bg" stroke="#e6e6e6" strokeWidth="8" fill="transparent" r="34" cx="40" cy="40"></circle>
                    <circle 
                      id="overall-score-circle" 
                      className="progress-ring__circle" 
                      stroke="#3366CC" 
                      strokeWidth="8" 
                      fill="transparent" 
                      r="34" 
                      cx="40" 
                      cy="40" 
                      strokeDasharray="213.52" 
                      strokeDashoffset="0"
                    ></circle>
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span id="overall-score" className="text-2xl font-bold">{resultData.result?.overall_score?.toFixed(1)}</span>
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-500">Overall Score</div>
                  <div className="text-lg font-semibold">Solid Foundation</div>
                </div>
              </div>
            </div>
          </div>

          {/* Score Cards */}
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-8">
            {resultData.result?.modules_analyzed.map((module) => {
              const score = resultData.result.detailed_results[module].overall_score;
              const indicator = getScoreIndicator(score);
              
              return (
                <div key={module} className="bg-white rounded-lg shadow-sm p-4 flex flex-col items-center">
                  <div className="text-lg font-semibold mb-2">{module.charAt(0).toUpperCase() + module.slice(1)}</div>
                  <div className="flex items-center">
                    <div id={`${module}-score`} className="text-2xl font-bold mr-2">{score.toFixed(1)}</div>
                    <div className={`text-sm ${indicator.color}`}>
                      <i className={`fas fa-${indicator.icon}`}></i> {indicator.text}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Main Sections */}
          <div className="flex flex-col lg:flex-row gap-6">
            {/* Left Column */}
            <div className="w-full lg:w-8/12">
              {/* Tabs */}
              <div className="bg-white rounded-lg shadow-sm mb-6">
                <div className="border-b border-gray-200">
                  <nav className="flex">
                    <button 
                      className={`py-4 px-6 focus:outline-none ${activeTab === 'summary' ? 'border-b' : ''}`}
                      onClick={() => setActiveTab('summary')}
                    >
                      Executive Summary
                    </button>
                    <button 
                      className={`py-4 px-6 focus:outline-none ${activeTab === 'improvements' ? 'border-b' : ''}`}
                      onClick={() => setActiveTab('improvements')}
                    >
                      Improvement Priority
                    </button>
                    <button 
                      className={`py-4 px-6 focus:outline-none ${activeTab === 'roadmap' ? 'border-b' : ''}`}
                      onClick={() => setActiveTab('roadmap')}
                    >
                      Implementation Roadmap
                    </button>
                  </nav>
                </div>
                
                {/* Tab Content */}
                <div className="p-6">
                  {/* Summary Tab */}
                  <div id="summary-tab" className={`tab-content ${activeTab !== 'summary' ? 'hidden' : ''}`}>
                    <h2 className="text-xl font-bold mb-4">Executive Summary</h2>
                    <p className="text-gray-700 leading-relaxed">
                      {resultData.result?.summary}
                    </p>
                    
                    <div className="mt-8">
                      <h3 className="text-lg font-semibold mb-3">Key Strengths</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {getSummaryStrengths().map((strength, index) => (
                          <div key={index} className="bg-green-50 p-4 rounded-lg border border-green-100">
                            <div className="flex items-center mb-2">
                              <div className="w-8 h-8 rounded-full bg-green-100 text-green-600 flex items-center justify-center mr-2">
                                <CheckCircle className="h-4 w-4" />
                              </div>
                              <h4 className="font-semibold">{strength.title}</h4>
                            </div>
                            <p className="text-sm text-gray-600">{strength.description}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Improvements Tab */}
                  <div id="improvements-tab" className={`tab-content ${activeTab !== 'improvements' ? 'hidden' : ''}`}>
                    <div className="flex justify-between items-center mb-6">
                      <h2 className="text-xl font-bold">Improvement Priorities</h2>
                      <div className="flex space-x-2">
                        <button className="text-sm bg-gray-100 hover:bg-gray-200 px-3 py-1 rounded-md transition-colors duration-200">
                          Filter
                        </button>
                        <button className="text-sm bg-gray-100 hover:bg-gray-200 px-3 py-1 rounded-md transition-colors duration-200">
                          Sort
                        </button>
                      </div>
                    </div>
                    
                    <div className="space-y-4">
                      {/* Priority 5 items (most important) */}
                      {getPriorityItems(5)}
                      
                      {/* Priority 4 items */}
                      {getPriorityItems(4)}
                      
                      {/* Priority 3 items */}
                      {getPriorityItems(3)}
                    </div>
                  </div>

                  {/* Roadmap Tab */}
                  <div id="roadmap-tab" className={`tab-content ${activeTab !== 'roadmap' ? 'hidden' : ''}`}>
                    <h2 className="text-xl font-bold mb-6">Implementation Roadmap</h2>
                    
                    <div className="relative">
                      {/* Immediate Actions */}
                      <div className="mb-12 ml-6 timeline-dot">
                        <div className="flex items-center mb-4">
                          <div className="absolute -left-2 w-6 h-6 rounded-full bg-red-500 flex items-center justify-center">
                            <AlertTriangle className="text-white h-3 w-3" />
                          </div>
                          <h3 className="text-lg font-semibold">Immediate Actions</h3>
                        </div>
                        
                        <div className="space-y-4">
                          {resultData.result?.implementation_roadmap.immediate_actions.map((action, index) => (
                            <div key={`immediate-${index}`} className="bg-white rounded-lg border border-gray-200 p-4">
                              <div className="flex justify-between items-start">
                                <div>
                                  <div className="text-gray-500 text-sm">{action.module}</div>
                                  <h4 className="font-medium">{action.task}</h4>
                                </div>
                                <span className="bg-red-100 text-red-800 text-xs font-medium px-2.5 py-0.5 rounded-full">Priority 5</span>
                              </div>
                              <p className="text-sm text-gray-600 mt-2">{action.recommendation}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      {/* Short Term */}
                      <div className="mb-12 ml-6 timeline-dot">
                        <div className="flex items-center mb-4">
                          <div className="absolute -left-2 w-6 h-6 rounded-full bg-amber-500 flex items-center justify-center">
                            <Clock className="text-white h-3 w-3" />
                          </div>
                          <h3 className="text-lg font-semibold">Short Term</h3>
                        </div>
                        
                        <div className="space-y-4">
                          {resultData.result?.implementation_roadmap.short_term.map((action, index) => (
                            <div key={`short-${index}`} className="bg-white rounded-lg border border-gray-200 p-4">
                              <div className="flex justify-between items-start">
                                <div>
                                  <div className="text-gray-500 text-sm">{action.module}</div>
                                  <h4 className="font-medium">{action.task}</h4>
                                </div>
                                <span className="bg-amber-100 text-amber-800 text-xs font-medium px-2.5 py-0.5 rounded-full">Priority 4</span>
                              </div>
                              <p className="text-sm text-gray-600 mt-2">{action.recommendation}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      {/* Medium Term */}
                      <div className="mb-12 ml-6 timeline-dot">
                        <div className="flex items-center mb-4">
                          <div className="absolute -left-2 w-6 h-6 rounded-full bg-blue-500 flex items-center justify-center">
                            <Calendar className="text-white h-3 w-3" />
                          </div>
                          <h3 className="text-lg font-semibold">Medium Term</h3>
                        </div>
                        
                        <div className="space-y-4">
                          {resultData.result?.implementation_roadmap.medium_term.map((action, index) => (
                            <div key={`medium-${index}`} className="bg-white rounded-lg border border-gray-200 p-4">
                              <div className="flex justify-between items-start">
                                <div>
                                  <div className="text-gray-500 text-sm">{action.module}</div>
                                  <h4 className="font-medium">{action.task}</h4>
                                </div>
                                <span className="bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded-full">Priority 3</span>
                              </div>
                              <p className="text-sm text-gray-600 mt-2">{action.recommendation}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      {/* Future Considerations */}
                      {resultData.result?.implementation_roadmap.future_considerations && 
                       resultData.result?.implementation_roadmap.future_considerations.length > 0 && (
                        <div className="ml-6 timeline-dot">
                          <div className="flex items-center mb-4">
                            <div className="absolute 2 w-6 h-6 rounded-full bg-gray-500 flex items-center justify-center">
                              <Lightbulb className="text-white h-3 w-3" />
                            </div>
                            <h3 className="text-lg font-semibold">Future Considerations</h3>
                          </div>
                          
                          <div className="space-y-4">
                            {resultData.result?.implementation_roadmap.future_considerations.map((action, index) => (
                              <div key={`future-${index}`} className="bg-white rounded-lg border border-gray-200 p-4">
                                <div className="flex justify-between items-start">
                                  <div>
                                    <div className="text-gray-500 text-sm">{action.module}</div>
                                    <h4 className="font-medium">{action.task}</h4>
                                  </div>
                                  <span className="bg-gray-100 text-gray-800 text-xs font-medium px-2.5 py-0.5 rounded-full">Priority 2</span>
                                </div>
                                <p className="text-sm text-gray-600 mt-2">{action.recommendation}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Color Palette Preview */}
              <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
                <h2 className="text-xl font-bold mb-4">Color Analysis</h2>
                {resultData.result?.detailed_results.color?.cv_metrics && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    {resultData.result?.detailed_results.color?.cv_metrics.primary_colors.map((color, index) => (
                      <div key={index} className="flex flex-col items-center">
                        <div 
                          className={`w-16 h-16 rounded-lg shadow-sm ${color === '#FFFFFF' || color === '#F5F5F5' ? 'border border-gray-200' : ''}`} 
                          style={{ backgroundColor: color }}
                        ></div>
                        <div className="text-sm font-medium mt-2">{color}</div>
                        <div className="text-xs text-gray-500">
                          {index === 0 ? 'Primary' : index === 1 ? 'Background' : index === 2 ? 'Accent' : 'Text'}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h3 className="font-semibold mb-2">Contrast Issue Example</h3>
                  <div className="flex flex-col md:flex-row gap-4">
                    <div className="flex-1 p-4 bg-white rounded border border-gray-200">
                      <div className="font-medium mb-2">Current</div>
                      <div className="p-3 rounded" style={{ backgroundColor: '#FFFFFF' }}>
                        <span style={{ color: '#3366CC', fontWeight: 'normal' }}>Task Text Example</span>
                      </div>
                      <div className="text-sm text-gray-500 mt-2">
                        Contrast ratio: {resultData.result?.detailed_results.color?.cv_metrics?.contrast_ratio.toFixed(1) || '3.9'}:1 (Fails AA)
                      </div>
                    </div>
                    
                    <div className="flex-1 p-4 bg-white rounded border border-gray-200">
                      <div className="font-medium mb-2">Recommended</div>
                      <div className="p-3 rounded" style={{ backgroundColor: '#FFFFFF' }}>
                        <span style={{ color: '#306298', fontWeight: 'normal' }}>Task Text Example</span>
                      </div>
                      <div className="text-sm text-green-500 mt-2">Contrast ratio: 4.8:1 (Passes AA)</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Right Column */}
            <div className="w-full lg:w-4/12">
              {/* Skills Chart */}
              <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
                <h3 className="text-lg font-semibold mb-4">Module Analysis</h3>
                <canvas ref={radarChartRef} width="400" height="300"></canvas>
              </div>
              
              {/* Quick Stats */}
              <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
                <h3 className="text-lg font-semibold mb-4">Analysis Context</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Screen Type:</span>
                    <span className="font-medium">{resultData.result?.design_context.screen_type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Purpose:</span>
                    <span className="font-medium">{resultData.result?.design_context.estimated_purpose}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Complexity:</span>
                    <span className="font-medium capitalize">{resultData.result?.design_context.complexity_level}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Modules Analyzed:</span>
                    <span className="font-medium">{resultData.result?.modules_analyzed.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Issues Found:</span>
                    <span className="font-medium">{resultData.result?.improvement_priority.length}</span>
                  </div>
                </div>
              </div>
              
              {/* Critical Issues */}
              <div className="bg-white rounded-lg shadow-sm p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold">Critical Issues</h3>
                  <span className="bg-red-100 text-red-800 text-xs font-medium px-2.5 py-0.5 rounded-full">
                    Priority 5
                  </span>
                </div>
                
                <div className="space-y-4">
                  {resultData.result?.improvement_priority
                    .filter(item => item.priority === 5)
                    .slice(0, 2)
                    .map((item, index) => (
                      <div key={`critical-${index}`} className="p-3 bg-red-50 rounded-lg border border-red-100">
                        <div className="flex items-start">
                          <AlertTriangle className="h-5 w-5 text-red-500 mr-2 mt-0.5 flex-shrink-0" />
                          <div>
                            <h4 className="font-medium">{item.issue}</h4>
                            <p className="text-sm text-gray-600 mt-1">
                              Module: {item.module.charAt(0).toUpperCase() + item.module.slice(1)}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))
                  }
                </div>
                
                <button className="w-full mt-4 text-sm text-blue-600 hover:text-blue-800 flex items-center justify-center">
                  View all critical issues <ArrowRight className="ml-1 h-4 w-4" />
                </button>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default UiUxAnalysisDashboard;