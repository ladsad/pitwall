export default function WinnerCard({ driver }) {
  if (!driver) return null;

  return (
    <div className="flex flex-col h-full bg-[#000000]">
      <div className="px-4 py-3 border-b border-[#141414]">
        <h2 className="text-[9px] text-[#e10600] tracking-[0.14em] uppercase m-0">Predicted Winner</h2>
      </div>
      
      <div className="flex-1 flex flex-col items-center justify-center p-6">
        <div className="text-[80px] leading-none font-bold text-[#ffffff] mb-2 tracking-tighter">
          {driver.driver}
        </div>
        
        <div className="text-[#aaaaaa] text-sm uppercase tracking-widest mb-8 text-center">
          <div className="text-[#ffffff] font-bold mb-1">{driver.full_name || driver.driver}</div>
          <div className="text-[10px]">{driver.team}</div>
        </div>
        
        <div 
          style={{ backgroundColor: '#0d0000', border: '1px solid #2a0000' }}
          className="w-full py-4 flex flex-col items-center justify-center mb-6"
        >
          <span className="text-[10px] text-[#e10600] tracking-[0.14em] uppercase mb-1">Win Probability</span>
          <span className="text-4xl font-bold text-[#ffffff]">{(driver.win_probability * 100).toFixed(1)}%</span>
        </div>
        
        <div className="w-full grid grid-cols-2 gap-4">
          <div className="flex flex-col items-center">
            <span className="text-[9px] text-[#666666] uppercase tracking-widest mb-1">Uncertainty</span>
            <span className="text-[#e8e8e8] text-sm font-bold">
              {driver.uncertainty < 0.05 ? 'LOW' : driver.uncertainty < 0.15 ? 'MED' : 'HIGH'}
            </span>
          </div>
          <div className="flex flex-col items-center">
            <span className="text-[9px] text-[#666666] uppercase tracking-widest mb-1">Bootstrap N</span>
            <span className="text-[#e8e8e8] text-sm font-bold">1000</span>
          </div>
        </div>
      </div>
    </div>
  );
}
