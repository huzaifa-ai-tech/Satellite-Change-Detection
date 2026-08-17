import L from "leaflet";
import "leaflet-draw/dist/leaflet.draw.js";

// leaflet-draw 1.0.4's dist assigns `type` without declaring it inside
// readableArea, which throws ReferenceError in strict mode (ESM) as soon as a
// rectangle's area tooltip is rendered. The vite transform plugin cannot fix
// it because the dep is pre-bundled before transforms run, so patch the
// function at runtime here (fixes both dev and prod builds).
const defaultPrecision = {
  km: 2,
  ha: 2,
  m: 0,
  mi: 2,
  ac: 2,
  yd: 0,
  ft: 0,
  nm: 2,
};

L.GeometryUtil.readableArea = function (area, isMetric, precision) {
  var areaStr,
    units,
    prec = L.Util.extend({}, defaultPrecision, precision);

  if (isMetric) {
    units = ["ha", "m"];
    var type = typeof isMetric;
    if (type === "string") {
      units = [isMetric];
    } else if (type !== "boolean") {
      units = isMetric;
    }

    if (area >= 1000000 && units.indexOf("km") !== -1) {
      areaStr = L.GeometryUtil.formattedNumber(area * 0.000001, prec["km"]) + " km²";
    } else if (area >= 10000 && units.indexOf("ha") !== -1) {
      areaStr = L.GeometryUtil.formattedNumber(area * 0.0001, prec["ha"]) + " ha";
    } else {
      areaStr = L.GeometryUtil.formattedNumber(area, prec["m"]) + " m²";
    }
  } else {
    area /= 0.836127;

    if (area >= 3097600) {
      areaStr = L.GeometryUtil.formattedNumber(area / 3097600, prec["mi"]) + " mi²";
    } else if (area >= 4840) {
      areaStr = L.GeometryUtil.formattedNumber(area / 4840, prec["ac"]) + " acres";
    } else {
      areaStr = L.GeometryUtil.formattedNumber(area, prec["yd"]) + " yd²";
    }
  }

  return areaStr;
};

export default L.Draw;
export const Draw = L.Draw;
