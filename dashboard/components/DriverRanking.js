"use client";

import { useEffect, useState, useRef } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { getTeamColor } from '../lib/teamColors';

function AnimatedCounter({ value, duration }) {
  const [count, setCount] = useState(0);
  const shouldReduceMotion = useReducedMotion();

  useEffect(() => {
    if (shouldReduceMotion) {
      setCount(value);
      return;
    }
    
    let startTime;
    let animationFrame;
    
    const step = (timestamp) => {
      if (!startTime) startTime = timestamp;
      const progress = timestamp - startTime;
      const t = Math.min(progress / duration, 1);
      setCount(t * value);
      
      if (progress < duration) {
        animationFrame = requestAnimationFrame(step);
      }
    };
    
    animationFrame = requestAnimationFrame(step);
    
    return () => cancelAnimationFrame(animationFrame);
  }, [value, duration, shouldReduceMotion]);

  return <>{(count * 100).toFixed(1)}%</>;
}

export default function DriverRanking({ predictions, selectedDriver, onSelectDriver }) {
  const shouldReduceMotion = useReducedMotion();

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-[#141414]">
        <h2 className="text-[9px] text-[#e10600] tracking-[0.14em] uppercase m-0">Driver Ranking</h2>
      </div>
      <div className="flex-1 overflow-y-auto">
        <div style={{ backgroundColor: '#141414' }} className="grid gap-[1px]">
          {predictions.map((p, idx) => {
            const isActive = selectedDriver?.driver === p.driver;
            const isP1 = idx === 0;
            const teamColor = getTeamColor(p.team);
            
            return (
              <div 
                key={p.driver}
                onClick={() => onSelectDriver(p)}
                style={{ backgroundColor: '#000000', cursor: 'pointer' }}
                className="flex items-center px-4 py-2 hover:bg-[#050505] transition-colors relative"
              >
                <motion.div
                  initial={false}
                  animate={{ opacity: isActive ? 1 : 0 }}
                  transition={{ duration: 0.15 }}
                  className="absolute inset-0 pointer-events-none"
                  style={{ backgroundColor: '#0a0000' }}
                />

                <div className="w-8 text-right text-[#aaaaaa] text-sm mr-4 relative z-10">
                  {idx + 1}
                </div>
                
                <div className="flex items-center gap-2 w-24 relative z-10">
                  <div style={{ backgroundColor: teamColor, width: '3px', height: '13px' }}></div>
                  <span className="font-bold text-[#ffffff]">{p.driver}</span>
                </div>
                
                <div className="flex-1 flex items-center gap-3 relative z-10">
                  <div className="flex-1 h-1.5 bg-[#141414] overflow-hidden">
                    <motion.div 
                      initial={shouldReduceMotion ? { scaleX: p.win_probability } : { scaleX: 0 }}
                      animate={{ scaleX: p.win_probability }}
                      transition={{ 
                        delay: shouldReduceMotion ? 0 : idx * 0.05, 
                        duration: shouldReduceMotion ? 0 : 0.6, 
                        ease: [0.16, 1, 0.3, 1] 
                      }}
                      style={{ 
                        transformOrigin: "left",
                        width: '100%'
                      }}
                      className="h-full relative"
                    >
                      <div className="absolute inset-0" style={{ backgroundColor: '#444444' }}></div>
                      <motion.div
                        initial={false}
                        animate={{ opacity: isActive ? 1 : 0 }}
                        transition={{ duration: 0.15 }}
                        className="absolute inset-0"
                        style={{ backgroundColor: '#e10600' }}
                      ></motion.div>
                    </motion.div>
                  </div>
                  <div className="w-12 text-right text-sm text-[#e8e8e8]">
                    <AnimatedCounter value={p.win_probability} duration={800} />
                  </div>
                </div>
                
                <div className="w-20 text-right text-xs text-[#666666] ml-4 relative z-10">
                  ±{(p.uncertainty * 100).toFixed(1)}
                </div>
                
                <div className="w-8 text-right ml-2 text-xs relative z-10">
                  {p.trend === 'up' && <span className="text-green-500">▲</span>}
                  {p.trend === 'down' && <span className="text-red-500">▼</span>}
                  {p.trend === 'flat' && <span className="text-[#666666]">—</span>}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
