"use client";

import { useState, useEffect } from "react";
import TopBar from "../components/TopBar";
import HeroStrip from "../components/HeroStrip";
import DriverRanking from "../components/DriverRanking";
import WinnerCard from "../components/WinnerCard";
import SessionPanel from "../components/SessionPanel";
import FeatureImportance from "../components/FeatureImportance";
import HistoricalAccuracy from "../components/HistoricalAccuracy";
import Top3Summary from "../components/Top3Summary";

export default function Home() {
  const [data, setData] = useState(null);
  const [selectedDriver, setSelectedDriver] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("/predictions.json")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load predictions.json");
        return res.json();
      })
      .then((json) => {
        setData(json);
        if (json.predictions && json.predictions.length > 0) {
          setSelectedDriver(json.predictions[0]);
        }
      })
      .catch((err) => setError(err.message));
  }, []);

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen text-[#e10600]">
        Error: {error}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center min-h-screen text-[#aaaaaa]">
        Loading dashboard data...
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-[#000000]">
      <TopBar version={data.model_version} generatedAt={data.generated_at} />
      <HeroStrip 
        round={data.round} 
        event={data.event} 
        sessions={data.sessions_used} 
        accuracy={data.season_accuracy} 
      />

      <div style={{ backgroundColor: '#141414' }} className="flex-1 grid grid-cols-1 lg:grid-cols-[65%_35%] gap-[1px]">
        {/* Left Column */}
        <div className="flex flex-col gap-[1px]">
          <div className="h-[400px]">
            <DriverRanking 
              predictions={data.predictions} 
              selectedDriver={selectedDriver} 
              onSelectDriver={setSelectedDriver} 
            />
          </div>
          <div className="grid grid-cols-2 gap-[1px] flex-1">
            <FeatureImportance features={data.feature_importance} />
            <HistoricalAccuracy history={data.history} />
          </div>
        </div>

        {/* Right Column */}
        <div className="flex flex-col gap-[1px]">
          <div className="h-[250px]">
            <WinnerCard driver={selectedDriver} />
          </div>
          <div className="h-[150px]">
            <SessionPanel driver={selectedDriver} />
          </div>
          <div className="flex-1">
            <Top3Summary predictions={data.predictions} lambda={data.recency_lambda} />
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer style={{ backgroundColor: '#111111', borderTop: '1px solid #141414' }} className="py-2 px-4 flex justify-between items-center text-[9px] text-[#666666] tracking-widest uppercase">
        <div className="flex items-center gap-4">
          <span>PITWALL v0.1.0</span>
          <span>·</span>
          <span>DATA FastF1</span>
          <span>·</span>
          <span>MODEL GBTClassifier · Spark MLlib</span>
          <span>·</span>
          <span>DEPLOY Vercel · auto</span>
        </div>
        <div>NOT AFFILIATED WITH FIA OR FORMULA 1</div>
      </footer>
    </div>
  );
}
