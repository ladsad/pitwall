"use client";

import { useState, useEffect, useRef } from "react";
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
  const [events, setEvents] = useState([]);
  const [currentEvent, setCurrentEvent] = useState(null);
  const [currentModel, setCurrentModel] = useState("mae");
  const [isLoading, setIsLoading] = useState(false);
  const cache = useRef({});

  useEffect(() => {
    fetch("/api/events?season=2026")
      .then((res) => {
        if (!res.ok) throw new Error("API status " + res.status);
        return res.json();
      })
      .then((json) => {
        if (Array.isArray(json) && json.length > 0) {
          setEvents(json);
          // Default to the latest event
          setCurrentEvent(json[json.length - 1].event);
        } else {
          // Fallback if no events found yet
          setCurrentEvent("Monaco Grand Prix");
        }
      })
      .catch((err) => {
        console.error("Failed to load events:", err);
        setCurrentEvent("Monaco Grand Prix");
      });
  }, []);

  useEffect(() => {
    if (!currentEvent) return;

    const cacheKey = `${currentEvent}_${currentModel}`;
    if (cache.current[cacheKey]) {
      const cachedData = cache.current[cacheKey];
      setData(cachedData);
      if (cachedData.predictions && cachedData.predictions.length > 0) {
        setSelectedDriver(cachedData.predictions[0]);
      }
      return;
    }

    setIsLoading(true);
    setError(null);
    fetch(`/api/predictions?season=2026&event=${encodeURIComponent(currentEvent)}&model=${encodeURIComponent(currentModel)}`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load predictions.json");
        return res.json();
      })
      .then((json) => {
        cache.current[cacheKey] = json;
        setData(json);
        if (json.predictions && json.predictions.length > 0) {
          setSelectedDriver(json.predictions[0]);
        }
      })
      .catch((err) => setError(err.message))
      .finally(() => setIsLoading(false));
  }, [currentEvent, currentModel]);

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
      
      {/* Selectors Header */}
      <div className="bg-[#0a0a0a] border-b border-[#222] px-6 py-3 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-6">
          <div className="flex flex-col">
            <label className="text-[10px] text-[#888] font-mono tracking-widest uppercase mb-1">Race Event</label>
            <select 
              className="bg-transparent text-white text-sm font-medium outline-none cursor-pointer hover:text-[#e10600] transition-colors appearance-none pr-4 py-1"
              value={currentEvent || ""}
              onChange={(e) => setCurrentEvent(e.target.value)}
            >
              {events.length > 0 ? (
                events.map((ev) => (
                  <option key={ev.event} value={ev.event} className="bg-[#111] text-white py-2 px-3">
                    Round {ev.round}: {ev.event}
                  </option>
                ))
              ) : (
                <option value={currentEvent} className="bg-[#111] text-white py-2 px-3">{currentEvent}</option>
              )}
            </select>
          </div>

          <div className="h-8 w-px bg-[#333]"></div>

          <div className="flex flex-col">
            <label className="text-[10px] text-[#888] font-mono tracking-widest uppercase mb-1">Telemetry Model</label>
            <select 
              className="bg-transparent text-[#aaa] text-sm font-medium outline-none cursor-pointer hover:text-[#e10600] transition-colors appearance-none pr-4 py-1"
              value={currentModel}
              onChange={(e) => setCurrentModel(e.target.value)}
            >
              <option value="mae" className="bg-[#111] text-white py-2 px-3">MAE (Deep Learning)</option>
              <option value="rf" className="bg-[#111] text-white py-2 px-3">Random Forest (Baseline)</option>
            </select>
          </div>
        </div>
        
        <div className="text-[10px] text-[#555] font-mono uppercase tracking-widest">
          {isLoading ? (
             <span className="text-[#e10600] animate-pulse">Syncing Telemetry...</span>
          ) : (
             <span>System Ready</span>
          )}
        </div>
      </div>

      <HeroStrip
        round={data.round}
        event={data.event}
        sessions={data.sessions_used}
        accuracy={data.season_accuracy}
      />

      <div style={{ backgroundColor: '#141414' }} className={`flex-1 grid grid-cols-1 lg:grid-cols-[65%_35%] gap-[1px] transition-opacity duration-300 ${isLoading ? "opacity-50 pointer-events-none" : "opacity-100"}`}>
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
            <FeatureImportance features={selectedDriver?.feature_importance?.length > 0 ? selectedDriver.feature_importance : data.feature_importance} />
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
