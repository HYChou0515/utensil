import { CanvasWidget } from "@projectstorm/react-canvas-core";
import React from "react";

import domain from "../../../domain/CanvasDomain";
import CanvasWrapper from "../../components/CanvasWrapper";

const FlowCanvas = () => {
  return (
    <CanvasWrapper onDrop={domain.onDrop} onDragOver={domain.onDragOver}>
      <CanvasWidget engine={domain.diagramEngine} className={"canvas-widget"} />
    </CanvasWrapper>
  );
};
export default FlowCanvas;
