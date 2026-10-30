export const palette = {input:'#699779',embedding:'#a78bc3',linear:'#708bc3',attention:'#cc9a52',norm:'#7caea4',add:'#a9957b',output:'#c47d86',custom:'#89939c'};
export function validateModel(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw Error('Mimari bir JSON nesnesi olmalı.');
  if (!Array.isArray(value.nodes) || !value.nodes.length || value.nodes.length > 160) throw Error('nodes alanında 1–160 katman bulunmalı.');
  if (!Array.isArray(value.edges) || value.edges.length > 600) throw Error('edges alanı bir liste olmalı; en fazla 600 bağlantı desteklenir.');
  const ids = new Set();
  const nodes = value.nodes.map((n,i) => {
    if (!n || typeof n.id !== 'string' || !n.id.trim() || n.id.length>100 || ids.has(n.id)) throw Error(`${i+1}. katmanın id değeri boş, çok uzun veya tekrar ediyor.`);
    ids.add(n.id);
    if (!Array.isArray(n.shape) || !n.shape.length || n.shape.length > 6 || n.shape.some(d => !Number.isSafeInteger(d) || d < 1 || d > 1e9)) throw Error(`${n.id}: shape, 1–6 pozitif tam sayı içermeli (her boyut en fazla 1 milyar).`);
    if (n.params != null && (!Number.isSafeInteger(n.params) || n.params<0)) throw Error(`${n.id}: params sıfır veya pozitif tam sayı olmalı.`);
    if (n.position != null && (!Array.isArray(n.position) || n.position.length!==3 || n.position.some(x=>!Number.isFinite(x)||Math.abs(x)>10000))) throw Error(`${n.id}: position, -10000 ile 10000 arasında üç sayı olmalı.`);
    if(n.values != null && (!Array.isArray(n.values)||n.values.length>100000||n.values.some(v=>!Number.isFinite(v)))) throw Error(`${n.id}: values en fazla 100000 sonlu sayı içermeli.`);
    return {...n,label:String(n.label??n.id).slice(0,120),type:String(n.type??'custom').slice(0,60),description:String(n.description??'').slice(0,2000)};
  });
  const edges=value.edges.map(e=>{
    if(!e || !ids.has(e.source)||!ids.has(e.target)) throw Error('Bağlantılardan biri var olmayan bir katmanı gösteriyor.');
    return {source:e.source,target:e.target};
  });
  return {name:String(value.name??'İsimsiz mimari').slice(0,120),description:String(value.description??'').slice(0,2000),nodes,edges};
}
// Longest-path layout for DAGs. Cyclic components retain input order.
export function layoutModel(model) {
  const ranks=new Map(model.nodes.map(n=>[n.id,0]));
  const incoming=new Map(model.nodes.map(n=>[n.id,0]));
  const outgoing=new Map(model.nodes.map(n=>[n.id,[]]));
  model.edges.forEach(e=>{incoming.set(e.target,incoming.get(e.target)+1);outgoing.get(e.source).push(e.target);});
  const queue=model.nodes.filter(n=>incoming.get(n.id)===0).map(n=>n.id); const seen=new Set();
  for(let i=0;i<queue.length;i++){const id=queue[i];seen.add(id);for(const target of outgoing.get(id)){ranks.set(target,Math.max(ranks.get(target),ranks.get(id)+1));incoming.set(target,incoming.get(target)-1);if(!incoming.get(target))queue.push(target);}}
  let next=Math.max(0,...ranks.values())+1;
  model.nodes.filter(n=>!seen.has(n.id)).forEach(n=>ranks.set(n.id,next++));
  const groups=new Map();model.nodes.forEach(n=>{const r=ranks.get(n.id);if(!groups.has(r))groups.set(r,[]);groups.get(r).push(n);});
  const max=Math.max(...ranks.values());const result=new Map();
  groups.forEach((nodes,rank)=>nodes.forEach((n,i)=>result.set(n.id,n.position??[(i-(nodes.length-1)/2)*15,(max/2-rank)*4.8,0])));
  return result;
}
