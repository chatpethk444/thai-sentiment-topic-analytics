export type Product = {
  id: string;
  name: string;
  sold: number;
  revenue: string;
  rating: number;
  category: string;
};

export const metrics = [
  { label: "Page Views", value: "16,431", change: "+15.5%", positive: true },
  { label: "Visitors", value: "6,225", change: "+8.4%", positive: true },
  { label: "Click", value: "2,832", change: "-10.5%", positive: false },
  { label: "Orders", value: "1,224", change: "+4.4%", positive: true },
];

export const profitData = [52, 44, 61, 57, 72, 64, 82, 74, 91, 78, 98, 88, 106, 95, 117, 109, 125, 114, 131, 121, 142, 136, 151, 145, 163, 155, 174, 166, 181, 172];

export const activeDays = [
  { day: "Mon", value: 58 },
  { day: "Tue", value: 92 },
  { day: "Wed", value: 71 },
  { day: "Thu", value: 64 },
  { day: "Fri", value: 83 },
  { day: "Sat", value: 47 },
  { day: "Sun", value: 39 },
];

export const customers = [
  { label: "Retailers", value: "3,482", percent: 56, icon: "store" },
  { label: "Distributors", value: "1,846", percent: 31, icon: "box" },
  { label: "Wholesalers", value: "806", percent: 13, icon: "users" },
];

export const products: Product[] = [
  { id: "#83009", name: "Hybrid Active Noise Cancelling Headphones", sold: 842, revenue: "$84,200", rating: 4.9, category: "Audio" },
  { id: "#83010", name: "Smart Fitness Watch Pro", sold: 716, revenue: "$64,440", rating: 4.8, category: "Wearables" },
  { id: "#83011", name: "Wireless Mechanical Keyboard", sold: 593, revenue: "$47,440", rating: 4.7, category: "Accessories" },
  { id: "#83012", name: "Ultra HD Smart Display", sold: 488, revenue: "$43,920", rating: 4.6, category: "Displays" },
  { id: "#83013", name: "Portable Bluetooth Speaker", sold: 431, revenue: "$30,170", rating: 4.8, category: "Audio" },
];

export const widgetOptions = [
  { title: "Visitors by Device", description: "Compare traffic across desktop, mobile, and tablet devices.", tags: ["Traffic", "Devices"], icon: "device" },
  { title: "Dashboard Overview", description: "Track your most important business metrics in one place.", tags: ["Overview", "KPI"], icon: "grid" },
  { title: "Orders Performance", description: "Monitor order volume, revenue, and conversion performance.", tags: ["Orders", "Revenue"], icon: "cart" },
  { title: "Trend Analysis", description: "Discover patterns and changes across your key performance data.", tags: ["Trends", "Analytics"], icon: "trend" },
  { title: "Customer Segmentation", description: "Understand customer groups and their contribution to growth.", tags: ["Customers", "Segments"], icon: "users" },
];
