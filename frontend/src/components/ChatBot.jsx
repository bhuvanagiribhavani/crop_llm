/**
 * ============================================================================
 * CHATBOT COMPONENT - AI Interface Assistant
 * ============================================================================
 * Floating help chatbot providing rule-based interface guidance.
 * Keyword-driven responses for upload, predict, NDVI, yield, insights, etc.
 */

import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Send, Bot, User } from 'lucide-react';

// Rule-based response map
const RESPONSES = [
  {
    keywords: ['upload'],
    answer:
      'To upload a satellite image, go to the "Upload Image" page from the sidebar. Click the upload zone or drag-and-drop a Sentinel-2 GeoTIFF image. Once uploaded, click "Run Prediction" to segment the crop map.',
  },
  {
    keywords: ['predict', 'prediction', 'segment'],
    answer:
      'The prediction process uses a U-Net deep learning model trained on Sentinel-2 LULC data. After uploading an image, the model classifies each pixel into one of 8 land-cover classes (e.g., cropland, forest, water). Results appear on the Crop Map, Statistics, NDVI, and Yield pages.',
  },
  {
    keywords: ['ndvi'],
    answer:
      'NDVI (Normalized Difference Vegetation Index) measures vegetation health on a scale from -1 to +1. Values above 0.4 indicate dense, healthy vegetation. The NDVI Analysis page shows the current NDVI value, a colour scale, vegetation classification, and an interpretation guide.',
  },
  {
    keywords: ['yield'],
    answer:
      'The Yield Estimation page displays four key metrics: Estimated Yield (tons), Crop Area (hectares), Yield per Hectare (tons/ha), and a Confidence percentage. These are derived from the segmentation and NDVI results.',
  },
  {
    keywords: ['insight', 'llm', 'ai insight'],
    answer:
      'The LLM Insights page generates AI-powered crop analysis. First run a prediction (or load the demo), then visit "LLM Insights" in the sidebar and click "Generate Insight". It provides crop type identification, NDVI assessment, yield estimate, and actionable recommendations.',
  },
  {
    keywords: ['database', 'store', 'saved', 'history'],
    answer:
      'Predictions are currently stored in-memory on the server. Each time you run a prediction or load the demo, the results are saved and can be retrieved by the LLM Insights page. A PostgreSQL database integration is planned for persistent storage.',
  },
  {
    keywords: ['navigation', 'sidebar', 'menu', 'navigate'],
    answer:
      'Use the sidebar on the left to navigate between pages: Dashboard (home), Upload Image, Crop Map, NDVI Analysis, Crop Statistics, Yield Estimation, LLM Insights, Reports, and Help & Support. On mobile, tap the hamburger icon to open the sidebar.',
  },
  {
    keywords: ['dark', 'theme', 'mode'],
    answer:
      'You can toggle dark mode from the Settings panel. Click the gear icon in the header to open Settings, then switch the theme. Your preference is saved in the browser.',
  },
  {
    keywords: ['demo'],
    answer:
      'To quickly explore the dashboard without uploading an image, click the "Load Demo" button on the Upload page. This loads a sample prediction with pre-computed crop map, statistics, NDVI, and yield data.',
  },
  {
    keywords: ['hello', 'hi', 'hey', 'help'],
    answer:
      'Hello! I\'m the AI Interface Assistant. I can help you with uploading images, running predictions, understanding NDVI, yield estimation, LLM insights, navigation, and more. Just type a keyword or question!',
  },
];

const DEFAULT_RESPONSE =
  'I\'m not sure about that. Please refer to the AI Insights page or upload a prediction. You can also ask me about: upload, predict, ndvi, yield, insight, database, or navigation.';

function getResponse(input) {
  const lower = input.toLowerCase().trim();
  for (const rule of RESPONSES) {
    if (rule.keywords.some((kw) => lower.includes(kw))) {
      return rule.answer;
    }
  }
  return DEFAULT_RESPONSE;
}

const ChatBot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: 'bot',
      text: 'Hi! I\'m the AI Interface Assistant. How can I help you navigate the dashboard?',
    },
  ]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom on new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed) return;

    const userMsg = { role: 'user', text: trimmed };
    const botMsg = { role: 'bot', text: getResponse(trimmed) };

    setMessages((prev) => [...prev, userMsg, botMsg]);
    setInput('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <>
      {/* ── Floating Button ── */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full bg-green-600 hover:bg-green-700 text-white shadow-lg flex items-center justify-center transition-all duration-200 hover:scale-110 focus:outline-none focus:ring-2 focus:ring-green-400 focus:ring-offset-2"
          aria-label="Open help chatbot"
        >
          <MessageCircle className="w-6 h-6" />
        </button>
      )}

      {/* ── Chat Panel ── */}
      {isOpen && (
        <div className="fixed bottom-6 right-6 z-50 w-80 sm:w-96 h-[28rem] flex flex-col rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 overflow-hidden transition-all duration-300 animate-slideUp">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 bg-green-600 text-white">
            <div className="flex items-center gap-2">
              <Bot className="w-5 h-5" />
              <span className="font-semibold text-sm">AI Interface Assistant</span>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="hover:bg-green-700 rounded-full p-1 transition-colors"
              aria-label="Close chatbot"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3 bg-gray-50 dark:bg-gray-900">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex gap-2 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {msg.role === 'bot' && (
                  <div className="flex-shrink-0 w-7 h-7 rounded-full bg-green-100 dark:bg-green-900 flex items-center justify-center mt-0.5">
                    <Bot className="w-4 h-4 text-green-600 dark:text-green-400" />
                  </div>
                )}
                <div
                  className={`max-w-[75%] px-3 py-2 rounded-xl text-sm leading-relaxed ${
                    msg.role === 'user'
                      ? 'bg-green-600 text-white rounded-br-sm'
                      : 'bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-200 border border-gray-200 dark:border-gray-700 rounded-bl-sm'
                  }`}
                >
                  {msg.text}
                </div>
                {msg.role === 'user' && (
                  <div className="flex-shrink-0 w-7 h-7 rounded-full bg-green-600 flex items-center justify-center mt-0.5">
                    <User className="w-4 h-4 text-white" />
                  </div>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="px-3 py-2 border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about upload, NDVI, yield..."
                className="flex-1 text-sm px-3 py-2 rounded-full border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-700 text-gray-800 dark:text-gray-200 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
              />
              <button
                onClick={handleSend}
                disabled={!input.trim()}
                className="p-2 rounded-full bg-green-600 hover:bg-green-700 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors focus:outline-none focus:ring-2 focus:ring-green-400"
                aria-label="Send message"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ChatBot;
