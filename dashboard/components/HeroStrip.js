export default function HeroStrip({ round, event, sessions, accuracy }) {
  return (
    <div style={{ backgroundColor: '#141414', borderBottom: '1px solid #141414' }} className="grid grid-cols-4 gap-[1px]">
      <div style={{ backgroundColor: '#000000' }} className="flex flex-col justify-center px-6 py-4">
        <span className="text-[#666666] text-[9px] tracking-[0.14em] uppercase mb-1">Round</span>
        <span style={{ color: '#e10600' }} className="text-3xl font-bold">{round.toString().padStart(2, '0')}</span>
      </div>
      <div style={{ backgroundColor: '#000000' }} className="flex flex-col justify-center px-6 py-4">
        <span className="text-[#666666] text-[9px] tracking-[0.14em] uppercase mb-1">Event</span>
        <span className="text-[#ffffff] text-lg font-bold truncate">{event}</span>
      </div>
      <div style={{ backgroundColor: '#000000' }} className="flex flex-col justify-center px-6 py-4">
        <span className="text-[#666666] text-[9px] tracking-[0.14em] uppercase mb-1">Sessions Ingested</span>
        <span className="text-[#e8e8e8] text-sm">{sessions.join(' · ')}</span>
      </div>
      <div style={{ backgroundColor: '#000000' }} className="flex flex-col justify-center px-6 py-4">
        <span className="text-[#666666] text-[9px] tracking-[0.14em] uppercase mb-1">Season Accuracy (Top 3)</span>
        <div className="flex items-end gap-2">
          <span className="text-[#ffffff] text-lg font-bold">{(accuracy.top3_pct * 100).toFixed(0)}%</span>
          <span className="text-[#666666] text-xs mb-0.5">({accuracy.races} races)</span>
        </div>
      </div>
    </div>
  );
}
