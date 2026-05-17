"use client";

import { motion, useReducedMotion } from 'framer-motion';

export default function HistoricalAccuracy({ history }) {
  const shouldReduceMotion = useReducedMotion();

  return (
    <div className="flex flex-col h-full bg-[#000000]">
      <div className="px-4 py-3 border-b border-[#141414]">
        <h2 className="text-[9px] text-[#e10600] tracking-[0.14em] uppercase m-0">Historical Accuracy</h2>
      </div>
      
      <div className="flex-1 overflow-y-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-[#141414]">
              <th className="px-4 py-2 text-[10px] text-[#666666] font-normal uppercase tracking-wider">Event</th>
              <th className="px-4 py-2 text-[10px] text-[#666666] font-normal uppercase tracking-wider">Pred P1</th>
              <th className="px-4 py-2 text-[10px] text-[#666666] font-normal uppercase tracking-wider">Act P1</th>
              <th className="px-4 py-2 text-[10px] text-[#666666] font-normal uppercase tracking-wider text-center">Top 3</th>
            </tr>
          </thead>
          <tbody>
            {history.map((h, i) => (
              <motion.tr 
                key={i} 
                initial={shouldReduceMotion ? { opacity: 1 } : { opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.4, delay: shouldReduceMotion ? 0 : i * 0.04 }}
                className="border-b border-[#141414] hover:bg-[#050505] transition-colors"
              >
                <td className="px-4 py-2 text-xs text-[#e8e8e8] truncate max-w-[120px]">{h.event}</td>
                <td className="px-4 py-2 text-xs font-bold text-[#aaaaaa]">{h.predicted}</td>
                <td className="px-4 py-2 text-xs font-bold text-[#aaaaaa]">{h.actual}</td>
                <td className="px-4 py-2 text-xs text-center">
                  {h.top3_hit ? (
                    <motion.div
                      initial={shouldReduceMotion ? { scale: 1 } : { scale: 1.3 }}
                      animate={{ scale: 1 }}
                      transition={{ duration: 0.2, delay: shouldReduceMotion ? 0 : (i * 0.04) + 0.1 }}
                      style={{ color: '#2ecc71', display: 'inline-block' }}
                    >✓</motion.div>
                  ) : (
                    <span style={{ color: '#e10600' }}>✗</span>
                  )}
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
