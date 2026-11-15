
    // Constants
    const DURATION_MS = 30000; // 30 seconds exact playback

    let viewer;
    let worldData = null;
    let timelineData = null;
    let gridBounds = { n: 34.150, s: 34.030, e: -118.440, w: -118.605 }; // Default fallback
    
    // Rendering collections
    let gridPolygons = [];
    let firePoints;
    let unitEntities = {};
    let activeFireState = new Map(); // cellIdx -> state

    async function init() {
      // 1. Fetch /world data
      try {
        const res = await fetch('/world?t=' + Date.now());
        worldData = await res.json();
      } catch (e) {
        console.error("Failed to load /world", e);
        alert("Backend not reachable. Ensure uvicorn is running on port 8000.");
        return;
      }
      
      // Update footer
      const f = document.getElementById('footer');
      f.innerHTML = `Model: ${worldData.meta.checkpoint} | Traffic: ${worldData.meta.traffic_mode}<br>Not for operational use.`;

      // 2. Init Cesium
      viewer = new Cesium.Viewer('cesiumContainer', {
        baseLayerPicker: false,
        timeline: false,
        animation: false,
        homeButton: false,
        navigationHelpButton: false,
        geocoder: false,
        sceneModePicker: false,
        infoBox: false,
        selectionIndicator: false
      });

      const padding = 0.05; // ~5km padding to ensure rotated grid is not cut off
      const viewRect = Cesium.Rectangle.fromDegrees(
        gridBounds.w - padding,
        gridBounds.s - padding,
        gridBounds.e + padding,
        gridBounds.n + padding
      );

      viewer.scene.globe.showGroundAtmosphere = false;
      viewer.scene.globe.enableLighting = true;
      viewer.scene.globe.depthTestAgainstTerrain = true;
      // Removed cartographicLimitRectangle in favor of clippingPlanes
      viewer.scene.skyBox.show = false; // Hide stars so buildings outside terrain blend in
      
      // Dark visual style
      viewer.scene.backgroundColor = Cesium.Color.fromCssColorString('#020202');

      // 3. Load 3D Terrain, Buildings, and Imagery
      try {
        viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({
          url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
          maximumLevel: 19,
          rectangle: viewRect
        }));
      } catch(ei) { console.warn('Imagery failed', ei); }

      try {
        const terrain = Cesium.Terrain.fromWorldTerrain();
        viewer.scene.setTerrain(terrain);
        
        // If terrain fails to load (e.g. invalid Ion token), the globe disappears.
        terrain.readyPromise.catch(e => {
          console.error('Terrain failed to load:', e);
          alert('ERROR: 3D Terrain failed to load. Falling back to a flat 2D map.');
          viewer.scene.terrainProvider = new Cesium.EllipsoidTerrainProvider();
        });
      } catch(et) { console.warn('Terrain failed', et); }
      
      try {
        const bldgs = await Cesium.createOsmBuildingsAsync();
        bldgs.maximumScreenSpaceError = 2; // Force high detail
        bldgs.maximumMemoryUsage = 8192;   // 8GB max memory
        bldgs.foveatedScreenSpaceError = false; 
        bldgs.cullWithChildrenBounds = false;
        bldgs.cullRequestsWhileMoving = false; // Don't cancel network requests when zooming
        bldgs.skipLevelOfDetail = true; // Skip intermediate LODs
        viewer.scene.primitives.add(bldgs);
        
        // Use exact geometric corners from backend
        const nw = Cesium.Cartesian3.fromDegrees(-118.62922, 34.11241);
        const ne = Cesium.Cartesian3.fromDegrees(-118.44000, 34.14986);
        const sw = Cesium.Cartesian3.fromDegrees(-118.60490, 34.03024);
        const se = Cesium.Cartesian3.fromDegrees(-118.41587, 34.06766);
        
        const centerCartesian = Cesium.Cartesian3.lerp(nw, se, 0.5, new Cesium.Cartesian3());
        
        // Strict distances for the rotated rectangle
        const strictHalfWidth = Cesium.Cartesian3.distance(nw, ne) / 2.0;
        const strictHalfHeight = Cesium.Cartesian3.distance(nw, sw) / 2.0;
        
        // Map extends beyond the grid
        const mapPlanes = [
          new Cesium.ClippingPlane(new Cesium.Cartesian3( 1.0,  0.0, 0.0), strictHalfWidth + 3000.0), // West
          new Cesium.ClippingPlane(new Cesium.Cartesian3(-1.0,  0.0, 0.0), strictHalfWidth + 2000.0), // East 
          new Cesium.ClippingPlane(new Cesium.Cartesian3( 0.0,  1.0, 0.0), strictHalfHeight + 6000.0), // South (shifted downwards/south)
          new Cesium.ClippingPlane(new Cesium.Cartesian3( 0.0, -1.0, 0.0), strictHalfHeight + 3000.0)  // North (shifted downwards/south)
        ];
        
        // Buildings are strictly bounded to the 8x4 grid, shrunk by 40 meters
        // to prevent edge buildings from poking past the green lines due to Earth curvature/projection warping
        const bldgPlanes = [
          new Cesium.ClippingPlane(new Cesium.Cartesian3( 1.0,  0.0, 0.0), strictHalfWidth - 40.0),
          new Cesium.ClippingPlane(new Cesium.Cartesian3(-1.0,  0.0, 0.0), strictHalfWidth - 40.0),
          new Cesium.ClippingPlane(new Cesium.Cartesian3( 0.0,  1.0, 0.0), strictHalfHeight - 40.0),
          new Cesium.ClippingPlane(new Cesium.Cartesian3( 0.0, -1.0, 0.0), strictHalfHeight - 40.0)
        ];
        
        let localTransform = Cesium.Transforms.eastNorthUpToFixedFrame(centerCartesian);
        
        // Calculate the true geographic rotation angle of the grid (relative to due East)
        const dLat = 34.14986 - 34.11241;
        const dLon = -118.44000 - -118.62922;
        const avgLat = Cesium.Math.toRadians(34.1311);
        const dx = dLon * Math.cos(avgLat);
        const dy = dLat;
        const rotationAngle = Math.atan2(dy, dx);
        
        const rotationQuat = Cesium.Quaternion.fromAxisAngle(Cesium.Cartesian3.UNIT_Z, rotationAngle);
        const rotationMat3 = Cesium.Matrix3.fromQuaternion(rotationQuat);
        const rotationMat4 = Cesium.Matrix4.fromRotationTranslation(rotationMat3, Cesium.Cartesian3.ZERO);
        localTransform = Cesium.Matrix4.multiply(localTransform, rotationMat4, new Cesium.Matrix4());
        
        const mapClippingCollection = new Cesium.ClippingPlaneCollection({
          planes: mapPlanes,
          modelMatrix: localTransform,
          edgeWidth: 0.0,
          unionClippingRegions: true
        });
        
        const bldgClippingCollection = new Cesium.ClippingPlaneCollection({
          planes: bldgPlanes,
          modelMatrix: localTransform,
          edgeWidth: 0.0,
          unionClippingRegions: true
        });
        
        bldgs.clippingPlanes = bldgClippingCollection;
        viewer.scene.globe.clippingPlanes = mapClippingCollection;
      } catch(eb) { console.warn('Buildings failed', eb); }
      // 5. Draw Zones
      drawZones();
      
      // 6. Setup fire collections
      firePoints = viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection());
      
      // 7. Go Home
      goHome();
      
      // Setup initial readiness UI
      updateReadiness(worldData.meta.roster);
    }

    function drawZones() {
      if (!worldData || !worldData.zones) return;
      
      worldData.zones.forEach(z => {
        // z.polygon is array of [lat, lon]
        const pts = [];
        z.polygon.forEach(p => { pts.push(p[1], p[0]); }); // lon, lat
        
        // Coverage color
        // coverage: 1.0 (good) -> green, 0.0 (bad) -> red
        const c = z.coverage;
        const color = new Cesium.Color(1.0 - c, c * 0.8, 0.2, 0.15);
        const outlineColor = new Cesium.Color(1.0 - c, c * 0.8, 0.2, 0.8);
        
        // Polygon fill shading removed per user request to keep natural terrain color.
        // Add outline polyline (GroundPrimitives don't support true outlines well)
        viewer.entities.add({
          polyline: {
            positions: Cesium.Cartesian3.fromDegreesArray([...pts, pts[0], pts[1]]),
            width: 2,
            material: outlineColor,
            clampToGround: true,
            classificationType: Cesium.ClassificationType.TERRAIN
          }
        });
        
        // Add label
        let text = `S${z.id}`;
        if (z.id === 18) text += "\
(AIR ONLY)";
        
        viewer.entities.add({
          position: Cesium.Cartesian3.fromDegrees(z.centroid[1], z.centroid[0]),
          label: {
            text: text,
            font: 'bold 16px Inter',
            fillColor: outlineColor.withAlpha(0.9),
            style: Cesium.LabelStyle.FILL,
            verticalOrigin: Cesium.VerticalOrigin.CENTER,
            horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
            disableDepthTestDistance: Number.POSITIVE_INFINITY, // Show through terrain
            scaleByDistance: new Cesium.NearFarScalar(1.0e3, 1.0, 2.0e4, 0.3)
          }
        });
      });
    }

    function goHome() {
      if (!viewer || !worldData) return;
      
      const cw = worldData.grid.corners;
      const centerLon = (cw.nw[1] + cw.se[1]) / 2;
      const centerLat = (cw.nw[0] + cw.se[0]) / 2;
      
      viewer.camera.flyTo({
        destination: Cesium.Cartesian3.fromDegrees(centerLon, centerLat - 0.1, 8000),
        orientation: {
          heading: Cesium.Math.toRadians(11), // Grid is rotated ~11 deg
          pitch: Cesium.Math.toRadians(-35),
          roll: 0
        },
        duration: 2.0
      });
    }
    
    function updateReadiness(readyDict) {
      const el = document.getElementById('fleet-readiness');
      el.innerHTML = '';
      const total = worldData.meta.roster;
      
      for (const [r, count] of Object.entries(readyDict)) {
        const tot = total[r] || count;
        const pct = (count / tot) * 100;
        const name = r.replace('_', ' ').toUpperCase();
        
        el.innerHTML += `
          <div class="ready-row">
            <span>${name} (${count}/${tot})</span>
            <div class="ready-bar-bg">
              <div class="ready-bar" style="width: ${pct}%"></div>
            </div>
          </div>
        `;
      }
    }

    async function startSimulation() {
      const btn = document.getElementById('btn-start');
      const spinner = document.getElementById('btn-spinner');
      
      btn.disabled = true;
      btn.querySelector('span').textContent = "Running...";
      spinner.style.display = "block";
      
      try {
        const res = await fetch('/simulate?scenario=anchor');
        timelineData = await res.json();
        
        btn.querySelector('span').textContent = "Replay";
        
        // Start playback
        playTimeline();
      } catch (e) {
        console.error("Simulation failed", e);
        alert("Simulation failed. Check console.");
        btn.querySelector('span').textContent = "Run Policy";
        btn.disabled = false;
      } finally {
        spinner.style.display = "none";
      }
    }
    
    // --- Playback Engine ---
    
    let playbackStartTime = 0;
    let playbackRAF = null;
    let lastRenderedTick = -1;
    
    function playTimeline() {
      if (!timelineData || !timelineData.ticks.length) return;
      
      if (playbackRAF) cancelAnimationFrame(playbackRAF);
      
      // Reset state
      firePoints.removeAll();
      activeFireState.clear();
      document.getElementById('policy-log').innerHTML = '';
      Object.values(unitEntities).forEach(e => viewer.entities.remove(e));
      unitEntities = {};
      
      // Load initial fire
      const initFire = timelineData.initial_fire;
      initFire.forEach(([idx, state]) => {
        activeFireState.set(idx, state);
      });
      renderFireFull();
      
      playbackStartTime = performance.now();
      lastRenderedTick = -1;
      
      // Show/setup banner based on scenario
      const delayInfo = timelineData.scenario.delay;
      if (delayInfo.ticks > 0) {
        const b = document.getElementById('delay-banner');
        document.getElementById('delay-desc').textContent = delayInfo.reason;
        b.classList.add('show');
      }
      
      playbackRAF = requestAnimationFrame(playbackLoop);
    }
    
    function playbackLoop(now) {
      const elapsed = now - playbackStartTime;
      let p = elapsed / DURATION_MS;
      
      if (p >= 1.0) {
        p = 1.0;
        document.getElementById('btn-start').disabled = false;
        document.getElementById('delay-banner').classList.remove('show');
      } else {
        playbackRAF = requestAnimationFrame(playbackLoop);
      }
      
      const frames = timelineData.ticks.length;
      const exactFrame = p * (frames - 1);
      const currentTick = Math.floor(exactFrame);
      const fraction = exactFrame - currentTick;
      
      // Update UI
      document.getElementById('progress-fill').style.width = `${p * 100}%`;
      
      // Apply tick diffs if we advanced
      if (currentTick > lastRenderedTick) {
        // Fast forward through any missed ticks
        for (let t = lastRenderedTick + 1; t <= currentTick; t++) {
          applyTick(t);
        }
        lastRenderedTick = currentTick;
      }
      
      // Interpolate unit positions for smooth movement
      if (currentTick < frames) {
        const td = timelineData.ticks[currentTick];
        renderUnits(td.units);
      }
    }
    
    function applyTick(tIdx) {
      const td = timelineData.ticks[tIdx];
      if (!td) return;
      
      // 1. HUD Metrics
      document.getElementById('time-display').textContent = td.sim_time;
      document.getElementById('tick-display').textContent = `T+${td.t}`;
      document.getElementById('val-active').textContent = td.active;
      document.getElementById('val-burned').textContent = td.burned;
      document.getElementById('val-lost').textContent = td.destroyed_total;
      document.getElementById('val-reward').textContent = td.cum.toFixed(2);
      document.getElementById('val-wind').textContent = td.wind.toFixed(1);
      document.getElementById('val-vs').textContent = td.value.toFixed(2);
      
      updateReadiness(td.ready);
      
      // 2. Banner Logic
      const b = document.getElementById('delay-banner');
      if (td.held) {
        const remaining = timelineData.scenario.delay.ticks - td.t;
        document.getElementById('delay-ticks').textContent = Math.max(0, remaining);
      } else {
        b.classList.remove('show');
      }
      
      // 3. Fire Diff
      const fireAdd = td.fire.add || [];
      const fireDel = td.fire.del || [];
      
      let changed = false;
      fireDel.forEach(idx => {
        activeFireState.delete(idx);
        changed = true;
      });
      fireAdd.forEach(([idx, state]) => {
        activeFireState.set(idx, state);
        changed = true;
      });
      
      if (changed) renderFireFull(); // Or partial update if optimized
      
      // 4. Policy Log
      if (td.actions && td.actions.length > 0) {
        // Group similar actions
        const grouped = {};
        td.actions.forEach(a => {
          const k = `${a.r} -> ${a.t} (S${a.z})`;
          grouped[k] = (grouped[k] || 0) + 1;
        });
        
        const logEl = document.getElementById('policy-log');
        for (const [k, count] of Object.entries(grouped)) {
          const str = count > 1 ? `${count}x ${k}` : k;
          logEl.innerHTML = `
            <div class="log-entry">
              <span class="log-time">${td.sim_time}</span>
              <span>${str}</span>
            </div>
          ` + logEl.innerHTML; // Prepend
        }
        
        // Keep log bounded
        while (logEl.children.length > 30) {
          logEl.removeChild(logEl.lastChild);
        }
      }
    }
    
    function renderFireFull() {
      firePoints.removeAll();
      const meta = worldData.grid;
      const w = meta.width;
      
      const states = worldData.meta.states;
      const colThreat = Cesium.Color.fromCssColorString('#ff7f0e');
      const colBlaze = Cesium.Color.fromCssColorString('#d62728');
      const colBurned = Cesium.Color.fromCssColorString('#3a2a20');
      
      // Extract affine
      const [a, b, c, d, e, f_aff] = worldData.grid.transform || [meta.cell_size_m, 0, meta.corners.nw[1], 0, -meta.cell_size_m, meta.corners.nw[0]]; 
      
      const cw = meta.corners;
      
      activeFireState.forEach((state, idx) => {
        const row = Math.floor(idx / w);
        const col = idx % w;
        
        // Exact bilinear interpolation to handle the 13.4-degree geographic rotation!
        const u = col / w;
        const v = row / meta.height;
        
        const latNorth = cw.nw[0] * (1 - u) + cw.ne[0] * u;
        const lonNorth = cw.nw[1] * (1 - u) + cw.ne[1] * u;
        
        const latSouth = cw.sw[0] * (1 - u) + cw.se[0] * u;
        const lonSouth = cw.sw[1] * (1 - u) + cw.se[1] * u;
        
        const lat = latNorth * (1 - v) + latSouth * v;
        const lon = lonNorth * (1 - v) + lonSouth * v;
        
        let color, size;
        if (state === states.threat) { color = colThreat; size = 12; }
        else if (state === states.blaze) { color = colBlaze; size = 15; }
        else if (state === states.burned) { color = colBurned; size = 10; }
        else return;
        
        // Subsample burned areas to keep rendering fast
        if (state === states.burned && Math.random() > 0.2) return;
        
        firePoints.add({
          position: Cesium.Cartesian3.fromDegrees(lon, lat),
          color: color,
          pixelSize: size,
          disableDepthTestDistance: 5000 // Don't clip into terrain too easily
        });
      });
    }
    
    const UNIT_COLORS = {
      'helicopter': Cesium.Color.AQUA,
      'water_team': Cesium.Color.DODGERBLUE,
      'trench_crew': Cesium.Color.ORANGE,
      'rescue_vehicle': Cesium.Color.LIME
    };
    
    function renderUnits(units) {
      if (!units) return;
      
      // Track which IDs were seen
      const seen = new Set();
      
      units.forEach(u => {
        seen.add(u.u);
        const rtype = u.u.split(':')[0];
        
        if (!unitEntities[u.u]) {
          // Create entity
          unitEntities[u.u] = viewer.entities.add({
            id: u.u,
            position: Cesium.Cartesian3.fromDegrees(u.lon, u.lat, u.alt),
            point: {
              pixelSize: 12,
              color: UNIT_COLORS[rtype] || Cesium.Color.WHITE,
              outlineColor: Cesium.Color.BLACK,
              outlineWidth: 2,
              disableDepthTestDistance: Number.POSITIVE_INFINITY
            },
            label: {
              text: rtype === 'helicopter' ? '🚁' : (rtype === 'water_team' ? '🚒' : '🚜'),
              font: '20px sans-serif',
              pixelOffset: new Cesium.Cartesian2(0, -20),
              disableDepthTestDistance: Number.POSITIVE_INFINITY
            }
          });
        } else {
          // Update position
          unitEntities[u.u].position = Cesium.Cartesian3.fromDegrees(u.lon, u.lat, u.alt);
        }
        
        // Update styling based on state
        const e = unitEntities[u.u];
        if (u.s === 'idle') {
          e.point.color = Cesium.Color.GRAY;
          e.show = false; // Hide idle units at station
        } else if (u.s === 'move') {
          e.point.color = Cesium.Color.YELLOW;
          e.show = true;
        } else if (u.s === 'work') {
          e.point.color = UNIT_COLORS[rtype];
          e.show = true;
        } else {
          e.show = true;
        }
      });
      
      // Remove unseen
      Object.keys(unitEntities).forEach(id => {
        if (!seen.has(id)) {
          viewer.entities.remove(unitEntities[id]);
          delete unitEntities[id];
        }
      });
    }

    // Start on load
    window.onload = async function() {
      if (typeof Cesium === 'undefined') {
        alert("Failed to load CesiumJS library! The globe will not render. Check your network or adblocker.");
        return;
      }
      
      // Cesium Tokens
      Cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI2ZGQ0ZjEwNC1jMjlhLTRlNGUtYjllNi0wNDdiYzcxNjkwZDUiLCJpZCI6NDYxNTk3LCJpc3MiOiJodHRwczovL2FwaS5jZXNpdW0uY29tIiwiYXVkIjoidW5kZWZpbmVkX2RlZmF1bHQiLCJpYXQiOjE3ODUyNzI5NjF9.oJ2oHETvVIw69GjOaAxuDiron7whJZf9rx4X3Hm6xNE';
      Cesium.GoogleMaps.defaultApiKey = 'AIzaSyCNM4xw2KhA-EdRWElXl3OX0_NHxB067jA';

      await init();
    };
  