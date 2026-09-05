"use client";

import { useEffect, useId, useState } from "react";

export const SENT_COLORS: Record<string,string> = { pos:"#A78BFA", neu:"#2DD4BF", neg:"#34D399", q:"#FB7185" };

export function Ring({pct,color,icon,id}:{pct:number;color:string;icon:string;id:string}) {
  const r=41, c=2*Math.PI*r, value=Math.max(0,Math.min(100,pct));
  const [animated,setAnimated]=useState(0);
  const uid=useId().replace(/:/g,"-");
  useEffect(()=>{const raf=requestAnimationFrame(()=>setAnimated(value));return()=>cancelAnimationFrame(raf)},[value]);
  const arc=animated/100*c, glowId=`neon-${id}-${uid}`;
  return <div className="relative h-[104px] w-[104px] shrink-0">
    <svg width="104" height="104" viewBox="0 0 104 104" aria-hidden>
      <defs><linearGradient id={`${glowId}-grad`} x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stopColor={color}/><stop offset="100%" stopColor={color} stopOpacity=".55"/></linearGradient><filter id={glowId}><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
      <circle className="kpi-ring-track" cx="52" cy="52" r={r} stroke="rgba(255,255,255,.075)" strokeWidth="9" fill="none"/>
      <circle className="kpi-ring-inner" cx="52" cy="52" r="34" stroke="rgba(255,255,255,.035)" strokeWidth="1" fill="none"/>
      <circle cx="52" cy="52" r={r} stroke={`url(#${glowId}-grad)`} strokeWidth="9" fill="none" strokeLinecap="round" strokeDasharray={`${arc.toFixed(1)} ${c.toFixed(1)}`} transform="rotate(-90 52 52)" filter={`url(#${glowId})`} style={{transition:"stroke-dasharray 700ms cubic-bezier(.2,.8,.2,1)"}}/>
      <text className="kpi-ring-icon" x="52" y="61" textAnchor="middle" fontSize="20" fill="#F5F7FF">{icon}</text>
    </svg>
    <span className="absolute -bottom-1 -right-3 rounded-full px-2 py-0.5 text-[10px] font-black" style={{color,background:`${color}18`,border:`1px solid ${color}35`}}>{animated.toFixed(0)}%</span>
  </div>
}

export function KpiCard({title,sub,num,foot,pct,color,icon,id}:{title:string;sub:string;num:string;foot:React.ReactNode;pct:number;color:string;icon:string;id:string}) {
  return <div className="lux-card group relative flex min-h-[150px] items-center justify-between gap-4 overflow-hidden p-5">
    <div className="pointer-events-none absolute -right-16 -top-16 h-32 w-32 rounded-full opacity-10 blur-2xl transition group-hover:opacity-20" style={{background:color}}/>
    <div className="relative min-w-0"><div className="flex items-center gap-2"><div className="lux-title text-[14px] kpi-title">{title}</div><span className="status-dot" style={{background:color,boxShadow:`0 0 10px ${color}`}}/></div><div className="lux-sub mt-1 text-[11px] uppercase tracking-[.08em]">{sub}</div><div className="kpi-value mt-1 text-[31px] font-extrabold tracking-tight tabular-nums kpi-number">{num}</div><div className="mt-2 text-[11px] leading-5 text-slate-400">{foot}</div></div>
    <Ring pct={pct} color={color} icon={icon} id={id}/>
  </div>
}
