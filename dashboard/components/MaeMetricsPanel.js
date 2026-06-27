"use client";

import { motion, useReducedMotion } from 'framer-motion';

export default function MaeMetricsPanel({ driver }) {
  const shouldReduceMotion = useReducedMotion();

  if (!driver) return null;

  const topFeature = driver.feature_importance && driver.feature_importance.length > 0 
    ? driver.feature_importance[0].feature.replace(/_/g, ' ').toUpperCase() 
    : "N/A";
    
  const formatTrend = (val) => {
    if (val === undefined || val === null) return "—";
    return (val > 0 ? "+" : "") + val.toFixed(2);
  };

  const metrics = [
    { label: "WIN PROB", value: `${(driver.win_probability * 100).toFixed(1)}%`, desc: "prediction" },
    { label: "UNCERTAINTY", value: driver.uncertainty?.toFixed(4) || "—", desc: "variance" },
    { label: "MOMENTUM", value: formatTrend(driver.trend?.value), desc: driver.trend?.label?.toUpperCase() || "FLAT" },
    { label: "EXP FINISH", value: `P${driver.predicted_position}`, desc: "position" },
    { label: "KEY FACTOR", value: topFeature, desc: "primary impact", isText: true },
    { label: "TEAM", value: driver.team, desc: "constructor", isText: true }
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
          MAE TELEMETRY PROFILE · {driver.driver}
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
            className="flex flex-col items-center justify-center p-2 relative overflow-hidden"
          >
            <span className="text-[10px] text-[#00e1ff] font-bold mb-2 text-center w-full truncate">
              {m.label}
            </span>
            <span className={`font-bold text-[#ffffff] mb-1 text-center w-full truncate ${m.isText ? 'text-sm' : 'text-xl'}`}>
              {m.value}
            </span>
            <span className="text-[9px] text-[#aaaaaa] truncate w-full text-center">
              {m.desc}
            </span>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
