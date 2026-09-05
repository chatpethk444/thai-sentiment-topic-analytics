"use client";
import { Icon, type IconName } from "./DashboardIcons";

const main = [["Dashboard","home"],["Orders","cart"],["Products","package"],["Customers","users"],["Stores","store"]] as [string,IconName][];

export default function Sidebar() {
 return <aside className="fixed inset-y-0 left-0 z-40 hidden w-64 flex-col border-r border-slate-200 bg-white px-4 py-5 dark:border-slate-800 dark:bg-slate-950 lg:flex">
   <div className="flex items-center gap-2 px-3 pb-7"><span className="grid h-9 w-9 place-items-center rounded-xl bg-blue-600 text-white shadow-lg shadow-blue-600/20"><Icon name="sparkle" size={19}/></span><span className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">Shopeers</span></div>
   <nav className="space-y-1">
    {main.map(([label,icon])=><div key={label} className={`flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition ${label==="Dashboard"?"bg-blue-50 text-blue-600 dark:bg-blue-500/10 dark:text-blue-400":"text-slate-500 hover:bg-slate-50 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-900 dark:hover:text-white"}`}><Icon name={icon}/><span>{label}</span>{label==="Orders"&&<span className="ml-auto rounded-full bg-blue-100 px-2 py-0.5 text-xs text-blue-600 dark:bg-blue-500/15 dark:text-blue-300">46</span>}</div>)}
   </nav>
   <div className="mt-7 px-3 text-[11px] font-semibold uppercase tracking-widest text-slate-400">Workspace</div>
   <nav className="mt-2 space-y-1">
    {[['Finances','wallet'],['Analytics','chart']].map(([label,icon])=><div key={label} className="flex cursor-pointer items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium text-slate-500 transition hover:bg-slate-50 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-900 dark:hover:text-white"><Icon name={icon as IconName}/><span>{label}</span><Icon name="chevron" size={15} className="ml-auto"/></div>)}
   </nav>
   <div className="mt-auto space-y-1">
    <div className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium text-slate-500 dark:text-slate-400"><Icon name="settings"/>Settings</div>
    <div className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium text-slate-500 dark:text-slate-400"><Icon name="help"/>Help &amp; Support</div>
    <div className="mt-4 rounded-2xl bg-slate-900 p-4 text-white shadow-xl dark:bg-blue-600"><div className="mb-3 grid h-9 w-9 place-items-center rounded-xl bg-white/10"><Icon name="sparkle" size={18}/></div><p className="text-sm font-semibold">Upgrade to Premium!</p><p className="mt-1 text-xs leading-5 text-slate-300 dark:text-blue-100">Unlock advanced analytics and unlimited widgets.</p><button className="mt-4 w-full rounded-xl bg-white px-3 py-2 text-xs font-semibold text-slate-900 transition hover:bg-slate-100">Upgrade now</button></div>
   </div>
 </aside>
}
