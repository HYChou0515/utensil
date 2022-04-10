import colormap from "colormap";
import md5 from "md5";
import tinycolor from "tinycolor2";

const colors = [
  ...colormap({
    colormap: "cool",
    nshades: 1024,
    format: "hex",
    alpha: 1,
  }),
  ...colormap({
    colormap: "phase",
    nshades: 1024,
    format: "hex",
    alpha: 1,
  }),
  ...colormap({
    colormap: "plasma",
    nshades: 1024,
    format: "hex",
    alpha: 1,
  }),
  ...colormap({
    colormap: "viridis",
    nshades: 1024,
    format: "hex",
    alpha: 1,
  }),
];

export const strToColor = (s) => {
  const md5s = md5(s);
  const idx = Number(`0x${md5s.substring(0, 3)}`);
  return tinycolor(colors[idx]).brighten(10).lighten(10).toRgbString();
};
