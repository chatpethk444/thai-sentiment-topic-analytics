export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Sentiment = "pos" | "neu" | "neg" | "q";

export interface TopicPoint {
  topic_id: number;
  total: number;
  pos: number;
  neu: number;
  neg: number;
  q: number;
}

export interface Summary {
  source: string;
  total: number;
  loaded: number;
  counts: Record<Sentiment, number>;
  overall_counts: Record<Sentiment, number>;
  pct: Record<Sentiment, number>;
  per_topic: TopicPoint[];
  topic_keywords: { topic_id: number; keywords: string }[];
  sentiments: string[];
  topics: number[];
}

export interface Message {
  texts: string;
  category?: string;
  predicted_sentiment: string;
  sentiment_score?: number;
  topic_id?: number;
  topic_keywords?: string;
}

function params(filters: {
  sentiments: string[];
  topics: number[];
  q: string;
}): string {
  const p = new URLSearchParams();
  filters.sentiments.forEach((s) => p.append("sentiment", s));
  filters.topics.forEach((t) => p.append("topic", String(t)));
  if (filters.q.trim()) p.set("q", filters.q.trim());
  return p.toString();
}

export async function fetchSummary(filters: {
  sentiments: string[];
  topics: number[];
  q: string;
}): Promise<Summary> {
  const res = await fetch(`${API_URL}/dashboard/summary?${params(filters)}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`summary ${res.status}: ${await res.text()}`);
  return res.json();
}

export async function fetchMessages(
  filters: { sentiments: string[]; topics: number[]; q: string },
  limit = 100,
): Promise<{ total: number; items: Message[] }> {
  const res = await fetch(
    `${API_URL}/dashboard/messages?${params(filters)}&limit=${limit}`,
    { cache: "no-store" },
  );
  if (!res.ok) throw new Error(`messages ${res.status}: ${await res.text()}`);
  return res.json();
}

export async function fetchHealth(): Promise<{ status: string; sentiment_model: string }> {
  const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error(`health ${res.status}`);
  return res.json();
}
