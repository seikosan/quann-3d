import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {validateModel,layoutModel,palette} from './model.js';
export class NeuralRenderer {
  constructor(container,{labels,onSelect=()=>{}}={}){
    this.container=container;this.labels=labels;this.onSelect=onSelect;this.items=new Map();this.links=[];
    this.scene=new THREE.Scene();this.camera=new THREE.PerspectiveCamera(38,1,.1,10000);
    this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,preserveDrawingBuffer:true});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));this.renderer.setClearColor(0xf4f5ef,0);container.append(this.renderer.domElement);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;this.controls.dampingFactor=.08;
    this.scene.add(new THREE.AmbientLight(0xffffff,1.3));const light=new THREE.DirectionalLight(0xffffff,1.6);light.position.set(-15,30,40);this.scene.add(light);
    this.root=new THREE.Group();this.scene.add(this.root);this.raycaster=new THREE.Raycaster();this.pointer=new THREE.Vector2();this.selected=null;
    this.down=e=>{this.start=[e.clientX,e.clientY];};this.up=e=>{if(!this.start||Math.hypot(e.clientX-this.start[0],e.clientY-this.start[1])>5)return;const rect=this.renderer.domElement.getBoundingClientRect();this.pointer.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);this.raycaster.setFromCamera(this.pointer,this.camera);const hits=this.raycaster.intersectObjects([...this.items.values()].map(v=>v.mesh));if(hits[0])this.select(hits[0].object.userData.id);};
    this.renderer.domElement.addEventListener('pointerdown',this.down);this.renderer.domElement.addEventListener('pointerup',this.up);
    this.resize=new ResizeObserver(()=>{const {width,height}=container.getBoundingClientRect();this.camera.aspect=width/height;this.camera.updateProjectionMatrix();this.renderer.setSize(width,height);});this.resize.observe(container);
    this.tick=()=>{this.frame=requestAnimationFrame(this.tick);this.controls.update();this.renderer.render(this.scene,this.camera);this.updateLabels();};this.tick();
  }
  clear(){this.root.traverse(o=>{o.geometry?.dispose();if(o.material)(Array.isArray(o.material)?o.material:[o.material]).forEach(m=>m.dispose());});this.root.clear();this.items.clear();this.links=[];if(this.labels)this.labels.replaceChildren();}
  load(data){const model=validateModel(data);this.clear();this.model=model;const positions=layoutModel(model);const dummy=new THREE.Object3D();
    model.nodes.forEach(n=>{
      const last=n.shape.at(-1),before=n.shape.at(-2)??1;
      const cols=Math.min(last,22),rows=Math.min(before,10),depth=n.shape.length>=3?Math.min(n.shape.slice(0,-2).reduce((a,b)=>a*b,1),3):1;
      const w=4.4+Math.log2(last+1)*.8,h=.5+Math.log2(before+1)*.4,d=.28+(depth-1)*.35;
      const geometry=new THREE.BoxGeometry(w/cols*.85,h/rows*.82,.18);const material=new THREE.MeshStandardMaterial({color:0xffffff,roughness:.65,metalness:.05});
      const mesh=new THREE.InstancedMesh(geometry,material,cols*rows*depth);let index=0;const color=new THREE.Color(palette[n.type]??palette.custom);let max=n.values?.length?Math.max(...n.values.map(Math.abs)):1;max=max||1;
      for(let z=0;z<depth;z++)for(let y=0;y<rows;y++)for(let x=0;x<cols;x++){
        dummy.position.set((x-(cols-1)/2)*w/cols,(y-(rows-1)/2)*h/rows,(z-(depth-1)/2)*.5);dummy.updateMatrix();mesh.setMatrixAt(index,dummy.matrix);
        if(n.values?.length){const v=n.values[Math.floor(index/(cols*rows*depth)*n.values.length)];mesh.setColorAt(index,new THREE.Color(v>=0?'#397b63':'#b56960').lerp(new THREE.Color('#f0f0e6'),1-Math.abs(v)/max));}else mesh.setColorAt(index,color);
        index++;
      }
      mesh.position.fromArray(positions.get(n.id));mesh.userData.id=n.id;this.root.add(mesh);
      const border=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(w+.18,h+.18,d+.3)),new THREE.LineBasicMaterial({color:palette[n.type]??palette.custom,transparent:true,opacity:.35}));border.position.copy(mesh.position);this.root.add(border);
      let label;if(this.labels){label=document.createElement('span');label.className='node-tag';label.textContent=n.label;this.labels.append(label);}
      this.items.set(n.id,{mesh,border,node:n,label,width:w,height:h});
    });
    model.edges.forEach((e,index)=>{const a=this.items.get(e.source),b=this.items.get(e.target);const start=a.mesh.position.clone().add(new THREE.Vector3(0,-a.height/2-.15,0)),end=b.mesh.position.clone().add(new THREE.Vector3(0,b.height/2+.15,0));
      const gap=start.y-end.y,skip=gap>7||gap<=0;let curve;
      if(skip){const side=Math.max(start.x,end.x)+10+(index%3)*.4;curve=new THREE.CubicBezierCurve3(start,new THREE.Vector3(side,start.y,-.3),new THREE.Vector3(side,end.y,-.3),end);}else curve=new THREE.CubicBezierCurve3(start,start.clone().add(new THREE.Vector3(0,-1.5,0)),end.clone().add(new THREE.Vector3(0,1.5,0)),end);
      const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(50)),new THREE.LineBasicMaterial({color:skip?0x849475:0x98aa8d,transparent:true,opacity:.7}));this.root.add(line);
      const point=curve.getPoint(.88),direction=curve.getTangent(.88);const arrow=new THREE.Mesh(new THREE.ConeGeometry(.10,.28,6),new THREE.MeshBasicMaterial({color:0x9aa88c}));arrow.position.copy(point);arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),direction);this.root.add(arrow);this.links.push({line,source:e.source,target:e.target});
    });
    this.fit();this.select(model.nodes[0].id);return model;
  }
  fit(front=false){const box=new THREE.Box3().setFromObject(this.root),size=box.getSize(new THREE.Vector3()),center=box.getCenter(new THREE.Vector3());const aspect=this.container.clientWidth/this.container.clientHeight;const dist=Math.max(size.y*1.65,size.x/aspect*1.65,size.z*2,12);this.camera.position.copy(center).add(front?new THREE.Vector3(0,0,dist*1.3):new THREE.Vector3(dist*.34,dist*.10,dist*1.18));this.controls.target.copy(center);this.controls.minDistance=2;this.controls.maxDistance=dist*5;this.camera.far=Math.max(10000,dist*10);this.camera.updateProjectionMatrix();this.controls.update();}
  select(id){if(!this.items.has(id))return;this.selected=id;this.items.forEach((item,key)=>{const yes=key===id;item.border.material.opacity=yes?1:.3;item.border.material.color.set(yes?'#2c5137':palette[item.node.type]??palette.custom);item.mesh.material.emissive.set(yes?'#182513':'#000000');item.label?.classList.toggle('active',yes);});this.links.forEach(l=>{l.line.material.color.set(l.source===id||l.target===id?'#608952':'#bcc6b2');l.line.material.opacity=l.source===id||l.target===id?1:.5;});this.onSelect(this.items.get(id).node);}
  updateLabels(){this.items.forEach(item=>{if(!item.label)return;const p=item.mesh.position.clone().add(new THREE.Vector3(0,-item.height/2-.65,0)).project(this.camera);item.label.style.display=p.z>1||p.z< -1?'none':'block';item.label.style.left=(p.x+1)/2*this.container.clientWidth+'px';item.label.style.top=(-p.y+1)/2*this.container.clientHeight+'px';});}
  exportPNG(){this.renderer.render(this.scene,this.camera);return this.renderer.domElement.toDataURL('image/png');}
  dispose(){cancelAnimationFrame(this.frame);this.resize.disconnect();this.controls.dispose();this.clear();this.renderer.domElement.removeEventListener('pointerdown',this.down);this.renderer.domElement.removeEventListener('pointerup',this.up);this.renderer.dispose();this.renderer.domElement.remove();}
}
