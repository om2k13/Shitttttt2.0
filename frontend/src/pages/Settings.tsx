import { useEffect, useState } from 'react'
import { api } from '../lib/api'

export default function Settings(){
  const [cfg,setCfg] = useState<any>({})
  useEffect(()=>{ api.get('/api/config').then(r=>setCfg(r.data)) },[])
  return <div>
    <h2>Settings (Policy)</h2>
    <pre style={{background:'#f6f6f6', padding:12}}>{JSON.stringify(cfg, null, 2)}</pre>
    <p>Edit <code>.codereview-agent.yml</code> to change behavior.</p>
  </div>
}
