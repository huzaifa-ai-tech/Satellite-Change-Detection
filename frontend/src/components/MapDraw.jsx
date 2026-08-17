import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet-draw";

const MapDraw = forwardRef(function MapDraw({ onRegionSelect }, ref) {
  const map = useMap();
  const fgRef = useRef(null);
  const cbRef = useRef(onRegionSelect);
  cbRef.current = onRegionSelect;

  useEffect(() => {
    const fg = new L.FeatureGroup().addTo(map);
    fgRef.current = fg;

    const control = new L.Control.Draw({
      position: "topright",
      draw: {
        rectangle: {
          showArea: true,
          shapeOptions: { color: "#06b6d4", weight: 2 },
        },
        polygon: false,
        polyline: false,
        circle: false,
        circlemarker: false,
        marker: false,
      },
      edit: {
        featureGroup: fg,
        edit: {
          selectedPathOptions: {
            dashArray: "6, 6",
            color: "#06b6d4",
            fill: true,
            fillColor: "#06b6d4",
            fillOpacity: 0.1,
            maintainColor: false,
          },
        },
        remove: false,
      },
    });
    map.addControl(control);

    const pick = (layer) => {
      const b = layer.getBounds();
      cbRef.current({
        north: b.getNorth(),
        south: b.getSouth(),
        east: b.getEast(),
        west: b.getWest(),
      });
    };

    const onCreated = (e) => {
      fg.clearLayers();
      fg.addLayer(e.layer);
      pick(e.layer);
    };
    const onEdited = (e) => {
      e.layers.eachLayer(pick);
    };
    const onDeleted = () => {
      cbRef.current(null);
    };

    map.on(L.Draw.Event.CREATED, onCreated);
    map.on(L.Draw.Event.EDITED, onEdited);
    map.on(L.Draw.Event.DELETED, onDeleted);

    return () => {
      map.off(L.Draw.Event.CREATED, onCreated);
      map.off(L.Draw.Event.EDITED, onEdited);
      map.off(L.Draw.Event.DELETED, onDeleted);
      map.removeControl(control);
      map.removeLayer(fg);
    };
  }, [map]);

  useImperativeHandle(ref, () => ({
    clear() {
      if (fgRef.current) fgRef.current.clearLayers();
      cbRef.current(null);
    },
  }));

  return null;
});

export default MapDraw;