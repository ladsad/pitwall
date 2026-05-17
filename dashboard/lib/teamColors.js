export const TEAM_COLORS = {
  "Red Bull Racing": "#3671C6",
  "Ferrari":         "#E8002D",
  "McLaren":         "#FF8000",
  "Mercedes":        "#27F4D2",
  "Aston Martin":    "#00A19C",
  "Alpine":          "#FF87BC",
  "Williams":        "#64C4FF",
  "RB":              "#6692FF",
  "Kick Sauber":     "#52E252",
  "Haas":            "#B6BABD",
};

export function getTeamColor(teamName) {
  return TEAM_COLORS[teamName] ?? "#555555";
}
