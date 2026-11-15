// Just a dummy to check my javascript logic syntax
const carto = Cesium.Cartographic.fromDegrees(-118.5, 34.0);
const surfaceNormal = Cesium.Ellipsoid.WGS84.geodeticSurfaceNormalCartographic(carto);
const rayDirection = Cesium.Cartesian3.negate(surfaceNormal, new Cesium.Cartesian3());
const rayOrigin = Cesium.Cartesian3.fromDegrees(-118.5, 34.0, 5000.0);
const ray = new Cesium.Ray(rayOrigin, rayDirection);
const intersection = viewer.scene.globe.pick(ray, viewer.scene);
