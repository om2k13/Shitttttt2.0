import { useEffect, useState } from 'react'
import { api } from '../lib/api'

export default function Reports(){
  const [jobs,setJobs] = useState<any>({items:[]})
  const load = async()=>{
    const r = await api.get('/api/jobs?size=50'); setJobs(r.data)
  }
  useEffect(()=>{ load() }, [])
  return <div>
    <h2>Reports</h2>
    <pre style={{background:'#f6f6f6', padding:12}}>{JSON.stringify(jobs, null, 2)}</pre>
  </div>
}
