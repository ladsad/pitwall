"use client";

import { motion, useReducedMotion } from 'framer-motion';

export default function FeatureImportance({ features }) {
  const shouldReduceMotion = useReducedMotion();

  // Sort features by importance just in case
  const sortedFeatures = [...features].sort((a, b) => b.importance - a.importance);
  const maxVal = sortedFeatures.length > 0 ? sortedFeatures[0].importance : 1;

  // Simple function to interpolate between #660000 and #e10600 based on value
  const getColor = (ratio) => {
    // #e10600 -> rgb(225, 6, 0)
    // #660000 -> rgb(102, 0, 0)
    const r = Math.round(102 + (225 - 102) * ratio);
    const g = Math.round(0 + (6 - 0) * ratio);
    const b = 0;
    return `rgb(${r}, ${g}, ${b})`;
  };

  return (
    <div className="flex flex-col h-full bg-[#000000]">
      <div className="px-4 py-3 border-b border-[#141414]">
        <h2 className="text-[9px] text-[#e10600] tracking-[0.14em] uppercase m-0">Feature Importance</h2>
      </div>
      
      <div className="p-4 flex flex-col gap-3">
        {sortedFeatures.map((f, i) => {
          const ratio = f.importance / maxVal;
          const targetScale = Math.max(ratio, 0.01);

          return (
            <div key={i} className="flex flex-col gap-1">
              <div className="flex justify-between text-xs">
                <span className="text-[#aaaaaa]">{f.feature}</span>
                <span className="text-[#ffffff]">{f.importance.toFixed(3)}</span>
              </div>
              <div className="h-1.5 w-full bg-[#141414] overflow-hidden">
                <motion.div 
                  initial={shouldReduceMotion ? { scaleX: targetScale } : { scaleX: 0 }}
                  animate={{ scaleX: targetScale }}
                  transition={{ 
                    delay: shouldReduceMotion ? 0 : i * 0.06, 
                    duration: shouldReduceMotion ? 0 : 0.7, 
                    ease: [0.16, 1, 0.3, 1] 
                  }}
                  className="h-full"
                  style={{ 
                    transformOrigin: "left",
                    width: '100%',
                    backgroundColor: getColor(ratio)
                  }}
                ></motion.div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
