"use client";

import { motion, useReducedMotion } from 'framer-motion';

export default function MaeMetricsPanel({ driver }) {
  const shouldReduceMotion = useReducedMotion();

  // Mock DL metrics to fill the space
  const metrics = [
    { label: "LATENCY", value: "14ms", desc: "per batch" },
    { label: "PARAMS", value: "14.5M", desc: "active" },
    { label: "EMBED", value: "256", desc: "dim size" },
    { label: "ATTN", value: "8", desc: "heads" },
    { label: "MASK", value: "75%", desc: "ratio" },
    { label: "LOSS", value: "0.24", desc: "val_mse" }
  ];

  const getVariants = () => {
    if (shouldReduceMotion) return { hidden: { opacity: 1 }, visible: { opacity: 1 } };
    return {
      hidden: { opacity: 0, y: 6 },
      visible: { opacity: 1, y: 0, transition: { duration: 0.3, ease: "easeOut" } }
    };
  };

  return (
    <div className="flex flex-col h-full bg-[#000000]">
      <div className="px-4 py-3 border-b border-[#141414]">
        <h2 className="text-[9px] text-[#00e1ff] tracking-[0.14em] uppercase m-0">
          MAE ARCHITECTURE STATS
        </h2>
      </div>
      
      <div style={{ backgroundColor: '#141414' }} className="flex-1 grid grid-cols-3 md:grid-cols-6 gap-[1px] p-[1px]">
        {metrics.map((m, idx) => (
          <motion.div 
            key={m.label}
            initial="hidden"
            animate="visible"
            variants={getVariants()}
            transition={{ delay: !shouldReduceMotion ? idx * 0.03 : 0 }}
            style={{ backgroundColor: '#000000' }}
            className="flex flex-col items-center justify-center p-3 relative"
          >
            <span className="text-[10px] text-[#00e1ff] font-bold mb-2">
              {m.label}
            </span>
            <span className="text-xl font-bold text-[#ffffff] mb-1">
              {m.value}
            </span>
            <span className="text-[9px] text-[#aaaaaa]">
              {m.desc}
            </span>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
