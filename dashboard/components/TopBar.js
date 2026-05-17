export default function TopBar({ version, generatedAt }) {
  const dateObj = new Date(generatedAt);
  const dateStr = dateObj.toLocaleDateString('en-GB') + ' ' + dateObj.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });

  return (
    <div style={{ backgroundColor: '#111111', borderBottom: '1px solid #141414' }} className="flex justify-between items-center px-4 py-3 sticky top-0 z-50">
      <div className="flex items-center gap-1 font-bold text-lg tracking-wider">
        <span style={{ color: '#ffffff' }}>PIT</span>
        <span style={{ color: '#e10600' }}>WALL</span>
      </div>
      <div className="flex items-center gap-4 text-xs font-mono">
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-[#e10600]"></span>
          </span>
          <span style={{ color: '#e10600' }} className="font-bold">LIVE</span>
        </div>
        <div style={{ border: '1px solid #333', backgroundColor: '#000', borderRadius: '2px' }} className="px-2 py-0.5 text-[#aaaaaa]">
          {version}
        </div>
        <div className="text-[#666666]">
          UPDATED: {dateStr}
        </div>
      </div>
    </div>
  );
}
