"use strict";
/* Sagittal landmark model: schematic, illustrative scale (8 px = 1 mm). */
(function(){
const $=id=>document.getElementById(id),NS=8,n=251,dt=5/(n-1);
const LM={bn:{u:[.34,.94],K:10,b:[226,210]},arj:{u:[.6,.8],K:14,b:[428,262]}};
const txt={knack:'Voluntary contraction: the landmark moves cranially and anteriorly, toward the symphysis (Δy > 0), then relaxes. Magnitudes are illustrative.',cough:'Cough: a brief caudal dip followed by a rebound (Δy < 0 then > 0). Real responses vary with timing and support.',valsalva:'Valsalva: sustained caudal descent (Δy < 0) with velocity peaks at onset and release.'};
let D,play=0,raf=0;
const f=(v,d)=>(v>=0?'+':'−')+Math.abs(v).toFixed(d);
function build(){const mode=$('d-mode').value,L=LM[$('d-lm').value];
 const s=Array.from({length:n},(_,i)=>trace(i*dt,mode,1,1)),x=s.map(v=>L.K*L.u[0]*v),y=s.map(v=>L.K*L.u[1]*v),vx=derivative(x,dt),vy=derivative(y,dt);
 D={L,x,y,vx,vy,ax:derivative(vx,dt),ay:derivative(vy,dt)};
 const P=i=>[L.b[0]-D.x[i]*NS,L.b[1]-D.y[i]*NS];
 D.P=P;$('trail').setAttribute('d',D.x.map((_,i)=>(i?'L':'M')+P(i).map(q=>q.toFixed(1)).join(' ')).join(''));
 $('p0').setAttribute('cx',L.b[0]);$('p0').setAttribute('cy',L.b[1]);
 strip();$('diagram-caption').textContent=txt[mode]+' The red arrow is Δr = r(t) − r(t₀).';show()}
function strip(){const W=600,H=190,m=36;let o='';
 [['Δy(t), mm',D.y,'#7c3aed',0],['v_y(t), mm/s',D.vy,'#0891b2',1]].forEach(([nm,a,c,k])=>{const top=12+k*88,h=66,mx=Math.max(...a.map(Math.abs),.001)*1.15,Y=v=>top+h/2-v/mx*h/2,X=i=>m+i/(n-1)*(W-m-8);
  o+=`<line x1="${m}" x2="${W-8}" y1="${Y(0)}" y2="${Y(0)}" stroke="#cfd4ee"/><text x="44" y="${top+10}" font-size="10" fill="#56617c" font-family="system-ui">${nm}</text><text x="${m-4}" y="${top+2}" font-size="9" text-anchor="end" fill="#8a91ad">${mx.toFixed(0)}</text><path fill="none" stroke="${c}" stroke-width="2.4" d="${a.map((v,i)=>(i?'L':'M')+X(i).toFixed(1)+' '+Y(v).toFixed(1)).join('')}"/>`});
 o+=`<line id="cur" y1="6" y2="170" stroke="#e11d48" stroke-width="1.5"/><text x="${W-8}" y="186" font-size="10" text-anchor="end" fill="#56617c" font-family="system-ui">time, 0 to 5 s</text>`;$('strip').innerHTML=o}
function show(){const i=+$('dt').value,[px,py]=D.P(i),[bx,by]=D.L.b,dx=D.x[i],dy=D.y[i],mag=Math.hypot(dx,dy),v=Math.hypot(D.vx[i],D.vy[i]),a=Math.hypot(D.ax[i],D.ay[i]);
 const set=(id,o)=>Object.entries(o).forEach(([k,v])=>$(id).setAttribute(k,v));
 set('pt',{cx:px,cy:py});set('vec',{d:`M${bx} ${by}L${px} ${py}`});set('cmpx',{d:`M${bx} ${by}H${px}`});set('cmpy',{d:`M${px} ${by}V${py}`});
 set('l0',{x:bx+13,y:by+22});set('lt',{x:px+13,y:py-10});set('lv',{x:(bx+px)/2+10,y:(by+py)/2});set('lx',{x:(bx+px)/2-8,y:by+15});set('ly',{x:px+9,y:(by+py)/2+4});
 $('dtout').textContent='t = '+(i*dt).toFixed(2)+' s';
 const c=36+i/(n-1)*(600-36-8);$('cur').setAttribute('x1',c);$('cur').setAttribute('x2',c);
 $('dread').innerHTML=[['t',(i*dt).toFixed(2)+' s'],['Δr(t) = [Δx, Δy]','['+f(dx,1)+', '+f(dy,1)+'] mm'],['‖Δr‖',mag.toFixed(1)+' mm'],['v = [vx, vy]','['+f(D.vx[i],1)+', '+f(D.vy[i],1)+'] mm/s'],['‖v‖',v.toFixed(1)+' mm/s'],['‖a‖',a.toFixed(0)+' mm/s²']].map(r=>`<div><small>${r[0]}</small><b>${r[1]}</b></div>`).join('')}
function tick(){const el=$('dt');el.value=(+el.value+2)%251;show();raf=play?requestAnimationFrame(tick):0}
$('dplay').addEventListener('click',()=>{play=!play;$('dplay').textContent=play?'❚❚ Pause':'▶ Play';if(play&&!raf)raf=requestAnimationFrame(tick)});
$('dt').addEventListener('input',show);['d-mode','d-lm'].forEach(id=>$(id).addEventListener('change',build));build();
/* clinical tabs */
document.querySelectorAll('.ctabs button').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.ctabs button').forEach(x=>x.setAttribute('aria-selected',x===b));document.querySelectorAll('.cpanel').forEach(p=>p.hidden=p.id!=='tab-'+b.dataset.tab)}));
})();
