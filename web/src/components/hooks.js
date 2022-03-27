import * as SRD from "@projectstorm/react-diagrams";
import { useState } from "react";

export const useForceUpdate = () => {
  const [, setValue] = useState(0); // integer state
  return () => setValue((value) => value + 1); // update the state to force render
};

export const useDiagramEngine = () => {
  const diagramEngine = SRD.default();
  const activeModel = new SRD.DiagramModel();
  diagramEngine.setModel(activeModel);
  return diagramEngine;
};
