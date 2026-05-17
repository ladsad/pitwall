"use client";

import { motion, useReducedMotion } from 'framer-motion';

export default function SessionPanel({ driver }) {
  const shouldReduceMotion = useReducedMotion();

  if (!driver || !driver.sessions) return null;

  const sessionKeys = ['FP1', 'FP2', 'FP3', 'Q', 'SPR', 'SQ'];

  const getVariants = (hasData) => {
    if (shouldReduceMotion || !hasData) {
      return {
        hidden: { opacity: hasData ? 1 : 0.3, y: 0 },
        visible: { opacity: hasData ? 1 : 0.3, y: 0 }
      };
    }
    return {
      hidden: { opacity: 0, y: 6 },
      visible: { opacity: 1, y: 0, transition: { duration: 0.3, ease: "easeOut" } }
    };
  };

  return (
    <div className="flex flex-col h-full bg-[#000000]">
      <div className="px-4 py-3 border-b border-[#141414]">
        <h2 className="text-[9px] text-[#e10600] tracking-[0.14em] uppercase m-0">
          SESSION CONTRIBUTIONS · {driver.driver}
        </h2>
      </div>
      
      <div style={{ backgroundColor: '#141414' }} className="flex-1 grid grid-cols-3 md:grid-cols-6 gap-[1px] p-[1px]">
        {sessionKeys.map((key, idx) => {
          const session = driver.sessions[key];
          const isQ = key === 'Q';
          const hasData = session !== null && session !== undefined;
          
          return (
            <motion.div 
              key={`${driver.driver}-${key}`}
              initial="hidden"
              animate="visible"
              variants={getVariants(hasData)}
              transition={{ delay: hasData && !shouldReduceMotion ? idx * 0.03 : 0 }}
              style={{ 
                backgroundColor: '#000000',
                borderColor: isQ && hasData ? '#e10600' : 'transparent',
                borderWidth: '1px',
                borderStyle: 'solid'
              }}
              className="flex flex-col items-center justify-center p-3 relative"
            >
              <span 
                style={{ color: isQ && hasData ? '#e10600' : '#666666' }} 
                className="text-[10px] font-bold mb-2"
              >
                {key}
              </span>
              
              <span className="text-xl font-bold text-[#ffffff] mb-1">
                {hasData ? session.score.toFixed(1) : '—'}
              </span>
              
              <span className="text-[9px] text-[#aaaaaa]">
                {hasData ? `w=${session.weight.toFixed(2)}` : '—'}
              </span>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
