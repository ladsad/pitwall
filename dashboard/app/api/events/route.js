import { createClient } from "@supabase/supabase-js";

export const dynamic = 'force-dynamic';

export async function GET(request) {
  try {
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

    if (!supabaseUrl || !supabaseKey) {
      return Response.json({ error: "Supabase credentials missing" }, { status: 501 });
    }

    const supabase = createClient(supabaseUrl, supabaseKey);
    const { searchParams } = new URL(request.url);
    const season = parseInt(searchParams.get("season") || "2026", 10);

    // Get distinct events from predictions table
    const { data, error } = await supabase
      .from("predictions")
      .select("event, round")
      .eq("season", season);

    if (error) {
      return Response.json({ error: error.message }, { status: 502 });
    }

    // Deduplicate
    const uniqueEvents = [];
    const seen = new Set();
    data.forEach((row) => {
      if (!seen.has(row.event)) {
        seen.add(row.event);
        uniqueEvents.push({ event: row.event, round: row.round });
      }
    });

    // Sort by round
    uniqueEvents.sort((a, b) => a.round - b.round);

    return Response.json(uniqueEvents);
  } catch (error) {
    return Response.json({ error: "Unhandled API Error" }, { status: 504 });
  }
}
