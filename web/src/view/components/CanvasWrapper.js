import styled from "@emotion/styled";
import React from "react";

const Background = styled.div`
  height: 100%;
  background-color: ${(p) => p.background};
  background-size: 50px 50px;
  display: flex;
  > * {
    height: 100%;
    min-height: 100%;
    width: 100%;
  }
  background-image: radial-gradient(
    circle at 5px 5px,
    ${(p) => p.color} 2px,
    #ffffff00 2px,
    #ffffff00 1px
  );
`;
const CanvasDropzone = styled.div`
  flex-grow: 1;
  position: relative;
  cursor: move;
  overflow: hidden;
`;

const CanvasWrapper = (prop) => (
  <Background background={"rgb(240, 240, 240)"} color={"rgba(60,60,60,0.25)"}>
    <CanvasDropzone onDrop={prop.onDrop} onDragOver={prop.onDragOver}>
      {prop.children}
    </CanvasDropzone>
  </Background>
);

export default CanvasWrapper;
