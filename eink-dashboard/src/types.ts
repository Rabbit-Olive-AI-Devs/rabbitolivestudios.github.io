export interface Env {
  CACHE: KVNamespace;
  AI: Ai;
  IMAGES: any;
  PHOTOS: R2Bucket;
  TEST_AUTH_KEY?: string;
  FOOTBALL_DATA_KEY?: string;
}

// --- Moment Before types ---

export interface MomentBeforeData {
  year: number;
  location: string;
  title: string;
  scene: string;
  imagePrompt: string;
}

// --- Weather types ---

export interface WeatherLocation {
  zip: string;
  name: string;
  lat: number;
  lon: number;
  tz: string;
}

export interface WeatherCondition {
  code: number;
  label: string;
  icon: string;
}

export interface CurrentWeather {
  temp_c: number;
  feels_like_c: number;
  humidity_pct: number;
  wind_kmh: number;
  wind_dir_deg: number;
  wind_dir_label: string;
  wind_gusts_kmh: number;
  is_day: boolean;
  precip_mm_hr: number;
  condition: WeatherCondition;
}

export interface HourlyEntry {
  time: string;
  temp_c: number;
  precip_prob_pct: number;
  precip_mm: number;
  code: number;
  icon: string;
  is_day: boolean;
}

export interface DailyEntry {
  date: string;
  high_c: number;
  low_c: number;
  precip_prob_pct: number;
  precipitation_sum_mm: number;
  snowfall_sum_cm: number;
  code: number;
  icon: string;
  sunrise: string;
  sunset: string;
}

export interface NWSAlert {
  event: string;
  severity: string;
  headline: string;
  onset: string;
  expires: string;
}

export interface WeatherResponse {
  location: WeatherLocation;
  updated_at: string;
  current: CurrentWeather;
  hourly_12h: HourlyEntry[];
  daily_5d: DailyEntry[];
  precip_next_2h: number[];
  alerts: NWSAlert[];
  sunrise: string;
  sunset: string;
}

// --- Fact types ---

export interface FactPage {
  title: string;
  url: string;
}

export interface FactEvent {
  year: number;
  text: string;
  pages: FactPage[];
}

export interface FactResponse {
  date: string;
  display_date: string;
  event: FactEvent;
  source: string;
}

// --- Device types ---

export interface DeviceData {
  battery_level: number;       // 0-100
  battery_charging: boolean;
  indoor_temp_c: number;       // rounded to integer
  indoor_humidity_pct: number; // rounded to integer
}

// --- Headlines types ---

export interface Headline {
  title: string;
  source: string;
  timestamp: string;
  summary: string;
  category: "tariffs" | "markets" | "company" | "regulatory";
  link?: string;
}

// --- KV cache wrapper ---

export interface CachedValue<T> {
  data: T;
  timestamp: number;
  /** Which provider produced this weather data (diagnostics; weather only). */
  source?: "open-meteo" | "nws";
}

// --- World Cup 2026 types ---

export type WcStage = "GROUP" | "R32" | "R16" | "QF" | "SF" | "THIRD" | "FINAL";
export type WcStatus = "SCHEDULED" | "LIVE" | "FINISHED";
export type WcPhase = "group" | "r32" | "knockout" | "champion";

export interface WcTeam {
  name: string;
  code: string; // 3-letter (BRA, USA); "TBD" when undecided
}

export interface WcMatch {
  id: number;
  stage: WcStage;
  group?: string;            // "A".."L" for group stage
  status: WcStatus;
  kickoffISO: string;        // UTC ISO from source
  dateChicago: string;       // YYYY-MM-DD in America/Chicago
  timeChicago: string;       // "1:00 PM" in America/Chicago
  home: WcTeam;
  away: WcTeam;
  homeScore: number | null;
  awayScore: number | null;
}

export interface WcStandingRow {
  position: number;
  team: WcTeam;
  played: number;
  won: number;
  drawn: number;
  lost: number;
  goalDifference: number;
  points: number;
  qualifying: boolean;       // top 2 only; third place never auto-marked
}

export interface WcGroup {
  name: string;              // "A".."L"
  rows: WcStandingRow[];
}

export interface WorldCupData {
  source: "football-data" | "openfootball";
  phase: WcPhase;
  todayMatches: WcMatch[];
  recentResults: WcMatch[];  // most recent finished matchday
  groups: WcGroup[];         // [] once group stage is over
  knockout: WcMatch[];       // R32..Final matches
  champion: WcTeam | null;
  generatedAt: number;       // epoch ms
}
