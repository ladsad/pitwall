import { createClient } from "@supabase/supabase-js";

export const dynamic = 'force-dynamic';

export async function GET(request) {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseKey) {
    console.warn("Supabase credentials missing");
    return Response.json({ error: "Supabase credentials missing" }, { status: 500 });
  }

  const supabase = createClient(supabaseUrl, supabaseKey);
  const { searchParams } = new URL(request.url);
  const season = parseInt(searchParams.get("season") || "2026", 10);
  const event = searchParams.get("event");

  if (!event) {
    return Response.json({ error: "Missing 'event' query param" }, { status: 400 });
  }

  const [predictionsRes, historyRes] = await Promise.all([
    supabase
      .from("predictions")
      .select("*")
      .eq("season", season)
      .eq("event", event)
      .order("win_probability", { ascending: false }),
    supabase
      .from("race_history")
      .select("*")
      .eq("season", season)
      .order("round", { ascending: true }),
  ]);

  if (predictionsRes.error) {
    return Response.json({ error: predictionsRes.error.message }, { status: 500 });
  }

  const predictions = predictionsRes.data || [];
  const history = historyRes.data || [];

  // Find the most recent model version for this event
  const modelVersions = [...new Set(predictions.map((p) => p.model_version))];

  return Response.json({
    model_version: modelVersions[0] || "unknown",
    generated_at: predictions[0]?.generated_at || null,
    event,
    season,
    round: predictions[0]?.round || null,
    sessions_used: [],
    season_accuracy: {
      top3_pct: history.filter((h) => h.top3_hit).length / (history.length || 1),
      races: history.length,
    },
    recency_lambda: null,
    predictions: predictions.map((p) => ({
      driver: p.driver,
      team: p.team,
      predicted_position: p.predicted_position,
      win_probability: p.win_probability,
      uncertainty: p.uncertainty,
      trend: { label: "flat", value: null },
      sessions: {},
    })),
    feature_importance: [],
    history: history.map((h) => ({
      event: h.event,
      predicted: h.predicted,
      actual: h.actual,
      top3_hit: h.top3_hit,
    })),
  });
}
