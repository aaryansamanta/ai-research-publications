/* Palisades-focused Cesium client for the local InfernoTactics API. */
const API = '';
let viewer, state, buildings, roads, depots, fireEntities = new Map(), resourceEntities = new Map(), timer = null, queued = [], stepInFlight = false;
const scenarioPoints = {anchor:[207,222], mandeville:[57,371], getty:[162,451]};

async function getJson(path, options){ const r=await fetch(API+path, options); if(!r.ok) throw new Error(await r.text()); return r.json(); }
function el(id){return document.getElementById(id)}
function colorForFire(s){return s===3?Cesium.Color.RED.withAlpha(.85):Cesium.Color.ORANGE.withAlpha(.72)}
function pointToRectangle(p){const d=.00014;return Cesium.Rectangle.fromDegrees(p.lon-d,p.lat-d,p.lon+d,p.lat+d)}
function updateHud(s){
  el('tick').textContent=s.tick;el('reward').textContent=Number(s.total_reward).toFixed(0);el('destroyed').textContent=s.buildings_destroyed_total||0;el('fireCount').textContent=s.fire_cells.length;
  const terminal=el('terminal');terminal.style.display=s.done?'block':'none';terminal.textContent=s.contained?'CONTAINED':'TIMEOUT - FIRE NOT CONTAINED';
  el('fleet').innerHTML=Object.entries(s.available).map(([k,v])=>`<div><span>${k}</span><b>${v}</b></div>`).join('');
  el('events').innerHTML=(s.events||[]).slice(-8).reverse().map(e=>`<div>T${e.tick} ${e.dispatch?.map(d=>d.resource_type+':'+d.status).join(', ')||'no dispatch'} ${e.buildings_destroyed?'· '+e.buildings_destroyed+' buildings lost':''}</div>`).join('');
}
function renderFire(cells){
  const seen=new Set();
  for(const c of cells){const id=`${c.row}:${c.col}`;seen.add(id);let entity=fireEntities.get(id);const p={lat:c.lat,lon:c.lon};if(!entity){entity=viewer.entities.add({id:'fire-'+id,rectangle:{coordinates:pointToRectangle(p),material:colorForFire(c.state),height:15,extrudedHeight:c.state===3?65:30}});fireEntities.set(id,entity)}else entity.rectangle.material=colorForFire(c.state)}
  for(const [id,entity] of fireEntities){if(!seen.has(id)){viewer.entities.remove(entity);fireEntities.delete(id)}}
}
function renderResources(items){
  const colors={water_team:Cesium.Color.CYAN,trench_crew:Cesium.Color.GOLD,rescue_vehicle:Cesium.Color.LIME,helicopter:Cesium.Color.ORANGERED};
  for(const r of items){let e=resourceEntities.get(r.id);if(!e){e=viewer.entities.add({id:'resource-'+r.id,position:Cesium.Cartesian3.fromDegrees(r.position.lon,r.position.lat,r.position.height_m||15),point:{pixelSize:r.resource_type==='helicopter'?12:9,color:colors[r.resource_type]||Cesium.Color.WHITE,outlineColor:Cesium.Color.WHITE,outlineWidth:1},label:{text:r.resource_type,show:false,scale:.55,fillColor:colors[r.resource_type]||Cesium.Color.WHITE}});resourceEntities.set(r.id,e)}e.position=Cesium.Cartesian3.fromDegrees(r.position.lon,r.position.lat,r.position.height_m||15);e.point.color=colors[r.resource_type];e.label.show=r.state==='traveling'||r.state==='preparing'}
}
async function loadStatic(){
  const [b,r,d]=await Promise.all([getJson('/api/static/buildings'),getJson('/api/static/roads'),getJson('/api/static/depots')]);
  buildings=await Cesium.GeoJsonDataSource.load(b,{clampToGround:false});roads=await Cesium.GeoJsonDataSource.load(r,{clampToGround:true});depots=await Cesium.GeoJsonDataSource.load({type:'FeatureCollection',features:d.map(x=>({type:'Feature',geometry:{type:'Point',coordinates:[x.lon,x.lat]},properties:x}))},{clampToGround:true});
  viewer.dataSources.add(buildings);viewer.dataSources.add(roads);viewer.dataSources.add(depots);
  for(const x of buildings.entities.values){const height=Number(x.properties.height_m?.getValue?.()||0);x.polygon.height=0;x.polygon.extrudedHeight=Math.max(3,height);x.polygon.material=Cesium.Color.LIGHTGRAY.withAlpha(.55);x.polygon.outline=true;x.polygon.outlineColor=Cesium.Color.GRAY}
  for(const x of roads.entities.values){x.polyline.material=Cesium.Color.DARKSLATEGRAY.withAlpha(.5);x.polyline.width=1}
  for(const x of depots.entities.values){x.point={pixelSize:10,color:Cesium.Color.YELLOW};x.label={text:x.properties.name,show:true,scale:.45,fillColor:Cesium.Color.WHITE}}
}
async function reset(){stop();const key=el('scenario').value;const body=key==='anchor'||key==='mandeville'||key==='getty'?{ignition_point:scenarioPoints[key]}:{scenario:key==='multi'?'multi':'single'};state=await getJson('/api/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});fireEntities.forEach(e=>viewer.entities.remove(e));fireEntities.clear();updateHud(state);renderFire(state.fire_cells);renderResources(state.resources)}
async function step(){if(stepInFlight||state?.done)return;stepInFlight=true;try{const mode=el('mode').value;const body=(mode==='autopilot'||mode==='heuristic')?{mode}:{mode:'manual',actions:queued.splice(0)};state=await getJson('/api/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});updateHud(state);renderFire(state.fire_cells);renderResources(state.resources);if(state.done)stop()}catch(e){console.error(e);el('connection').textContent='ERROR'}finally{stepInFlight=false}}
function start(){if(!timer)timer=setInterval(step,700);el('connection').textContent='RUNNING'}function stop(){if(timer){clearInterval(timer);timer=null}el('connection').textContent='PAUSED'}
el('start').onclick=start;el('pause').onclick=stop;el('step').onclick=step;el('reset').onclick=reset;el('dispatch').onclick=()=>{queued.push([el('resource').value,Number(el('zone').value)]);el('connection').textContent=`QUEUED ${queued.length}`};el('camera').onclick=()=>viewer.camera.flyTo({destination:Cesium.Cartesian3.fromDegrees(-118.545,34.065,15000),orientation:{pitch:Cesium.Math.toRadians(-45)}});el('manual').style.display='none';el('mode').onchange=()=>el('manual').style.display=el('mode').value==='manual'?'block':'none';el('buildings').onchange=e=>buildings.show=e.target.checked;el('roads').onchange=e=>roads.show=e.target.checked;el('depots').onchange=e=>depots.show=e.target.checked;el('fire').onchange=e=>fireEntities.forEach(x=>x.show=e.target.checked);
(async function(){
  viewer=new Cesium.Viewer('cesiumContainer',{
    animation:false,
    timeline:false,
    // Use the explicit public OSM layer below. Keeping Cesium's Ion-backed
    // base-layer picker enabled makes Cesium request its default Ion asset and
    // leaves the globe black when no Ion token is configured.
    baseLayerPicker:false,
    geocoder:false,
    homeButton:true,
    navigationHelpButton:false,
    sceneModePicker:true,
    terrainProvider:new Cesium.EllipsoidTerrainProvider(),
    imageryProvider:false
  });
  viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({
    url:'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    credit:'© OpenStreetMap contributors'
  }));
  viewer.scene.globe.enableLighting=true;
  viewer.scene.globe.show=true;
  viewer.scene.backgroundColor=Cesium.Color.fromCssColorString('#071019');
  viewer.camera.setView({
    destination:Cesium.Rectangle.fromDegrees(-118.605,34.030,-118.485,34.105)
  });
  await loadStatic();
  await reset();
  viewer.camera.flyTo({
    destination:Cesium.Rectangle.fromDegrees(-118.605,34.030,-118.485,34.105),
    duration:0.8
  });
  el('connection').textContent='READY'
})().catch(e=>{console.error(e);el('connection').textContent='ERROR'})
