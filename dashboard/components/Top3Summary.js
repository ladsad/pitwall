export default function Top3Summary({ predictions, lambda }) {
  const top3 = predictions.slice(0, 3);
  
  // Calculate recency decays
  const decays = [
    { label: 'Last race', n: 1 },
    { label: '3 ago', n: 3 },
    { label: '10 ago', n: 10 }
  ];

  return (
    <div className="flex flex-col h-full bg-[#000000]">
      <div className="px-4 py-3 border-b border-[#141414]">
        <h2 className="text-[9px] text-[#e10600] tracking-[0.14em] uppercase m-0">Top-3 Summary</h2>
      </div>
      
      <div className="p-4 flex-1 flex flex-col justify-between">
        <div className="flex flex-col gap-2">
          {top3.map((p, idx) => (
            <div key={p.driver} className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="w-4 text-right text-xs text-[#666666]">P{idx + 1}</span>
                <span 
                  style={{ color: idx === 0 ? '#e10600' : '#aaaaaa' }}
                  className="font-bold text-sm"
                >
                  {p.driver}
                </span>
              </div>
              <div className="flex items-center gap-4 text-xs">
                <span className="text-[#ffffff]">{(p.win_probability * 100).toFixed(1)}%</span>
                <span className="w-10 text-right text-[#666666]">±{(p.uncertainty * 100).toFixed(1)}</span>
              </div>
            </div>
          ))}
        </div>
        
        <div className="mt-4 pt-4 border-t border-[#141414]">
          <div className="text-[9px] text-[#666666] tracking-widest uppercase mb-2">Recency Decay (λ={lambda})</div>
          <div className="flex justify-between text-xs text-[#aaaaaa]">
            {decays.map((d, i) => {
              const val = Math.exp(-lambda * d.n);
              return (
                <span key={i}>
                  {d.label}: <span className="text-[#ffffff]">{val.toFixed(2)}</span>
                  {i < decays.length - 1 && <span className="mx-1 text-[#444444]">·</span>}
                </span>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
