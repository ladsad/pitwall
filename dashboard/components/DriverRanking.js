import { getTeamColor } from '../lib/teamColors';

export default function DriverRanking({ predictions, selectedDriver, onSelectDriver }) {
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
                style={{ 
                  backgroundColor: isActive ? '#0a0000' : '#000000',
                  cursor: 'pointer'
                }}
                className="flex items-center px-4 py-2 hover:bg-[#050505] transition-colors"
              >
                <div className="w-8 text-right text-[#aaaaaa] text-sm mr-4">
                  {idx + 1}
                </div>
                
                <div className="flex items-center gap-2 w-24">
                  <div style={{ backgroundColor: teamColor, width: '3px', height: '13px' }}></div>
                  <span className="font-bold text-[#ffffff]">{p.driver}</span>
                </div>
                
                <div className="flex-1 flex items-center gap-3">
                  <div className="flex-1 h-1.5 bg-[#141414] overflow-hidden">
                    <div 
                      style={{ 
                        width: `${p.win_probability * 100}%`,
                        backgroundColor: isP1 ? '#e10600' : '#444444'
                      }}
                      className="h-full"
                    ></div>
                  </div>
                  <div className="w-12 text-right text-sm text-[#e8e8e8]">
                    {(p.win_probability * 100).toFixed(1)}%
                  </div>
                </div>
                
                <div className="w-20 text-right text-xs text-[#666666] ml-4">
                  ±{(p.uncertainty * 100).toFixed(1)}
                </div>
                
                <div className="w-8 text-right ml-2 text-xs">
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
