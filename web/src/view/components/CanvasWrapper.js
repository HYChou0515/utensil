import styled from "@emotion/styled";
import React, { useState } from "react";

import canvasDomain from "../../domain/CanvasDomain";

const CanvasDropzone = styled.div`
  flex-grow: 1;
  position: relative;
  cursor: move;
  overflow: hidden;
`;

const CanvasWrapper = (prop) => (
  <CanvasDropzone onDrop={prop.onDrop} onDragOver={prop.onDragOver}>
    {prop.children}
  </CanvasDropzone>
);

export default CanvasWrapper;
