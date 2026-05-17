export default function HistoricalAccuracy({ history }) {
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
              <tr key={i} className="border-b border-[#141414] hover:bg-[#050505] transition-colors">
                <td className="px-4 py-2 text-xs text-[#e8e8e8] truncate max-w-[120px]">{h.event}</td>
                <td className="px-4 py-2 text-xs font-bold text-[#aaaaaa]">{h.predicted}</td>
                <td className="px-4 py-2 text-xs font-bold text-[#aaaaaa]">{h.actual}</td>
                <td className="px-4 py-2 text-xs text-center">
                  {h.top3_hit ? (
                    <span style={{ color: '#2ecc71' }}>✓</span>
                  ) : (
                    <span style={{ color: '#e10600' }}>✗</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
