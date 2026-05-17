"use client";

import { motion, useReducedMotion } from 'framer-motion';

export default function HeroStrip({ round, event, sessions, accuracy }) {
  const shouldReduceMotion = useReducedMotion();

  const itemVariants = {
    hidden: { opacity: 0, y: shouldReduceMotion ? 0 : 8 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" } }
  };

  const cells = [
    { title: "Round", content: <span style={{ color: '#e10600' }} className="text-3xl font-bold">{round.toString().padStart(2, '0')}</span> },
    { title: "Event", content: <span className="text-[#ffffff] text-lg font-bold truncate">{event}</span> },
    { title: "Sessions Ingested", content: <span className="text-[#e8e8e8] text-sm">{sessions.join(' · ')}</span> },
    { title: "Season Accuracy (Top 3)", content: <div className="flex items-end gap-2"><span className="text-[#ffffff] text-lg font-bold">{(accuracy.top3_pct * 100).toFixed(0)}%</span><span className="text-[#666666] text-xs mb-0.5">({accuracy.races} races)</span></div> }
  ];

  return (
    <div style={{ backgroundColor: '#141414', borderBottom: '1px solid #141414' }} className="grid grid-cols-4 gap-[1px]">
      {cells.map((cell, idx) => (
        <motion.div 
          key={idx}
          initial="hidden"
          animate="visible"
          variants={itemVariants}
          transition={{ delay: idx * 0.08 }}
          style={{ backgroundColor: '#000000' }} 
          className="flex flex-col justify-center px-6 py-4"
        >
          <span className="text-[#666666] text-[9px] tracking-[0.14em] uppercase mb-1">{cell.title}</span>
          {cell.content}
        </motion.div>
      ))}
    </div>
  );
}
