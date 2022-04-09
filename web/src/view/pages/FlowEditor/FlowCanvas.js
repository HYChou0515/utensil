import { css } from "@emotion/react";
import styled from "@emotion/styled";
import {
  CanvasWidget,
  SmartLayerWidget,
  TransformLayerWidget,
} from "@projectstorm/react-canvas-core";
import React from "react";

import domain from "../../../domain/CanvasDomain";
import CanvasWrapper from "../../components/CanvasWrapper";

const shared = css`
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  position: absolute;
  pointer-events: none;
  transform-origin: 0 0;
  width: 100%;
  height: 100%;
  overflow: visible;
`;

const DivLayer = styled.div`
  ${shared}
`;

const SvgLayer = styled.div`
  background-color: rgb(240, 240, 240);
  background-size: 50px 50px;
  display: flex;
  > * {
    height: 100%;
    min-height: 100%;
    width: 100%;
  }
  background-image: radial-gradient(
    circle at 5px 5px,
    rgba(160, 60, 60, 0.25) 2px,
    #ffffff00 2px,
    #ffffff00 1px
  );
  ${shared}
`;

export class BackgroundLayerWidget extends React.Component {
  constructor(props) {
    super(props);
    this.state = {};
  }

  getTransform() {
    const model = this.props.layer.getParent();
    return `
			${model.getOffsetX()}px ${model.getOffsetY()}px
  	`;
  }

  getTransformStyle() {
    if (this.props.layer.getOptions().transformed) {
      return {
        backgroundPosition: this.getTransform(),
      };
    }
    return {};
  }

  render() {
    if (this.props.layer.getOptions().isSvg) {
      return (
        <SvgLayer style={this.getTransformStyle()}>
          {this.props.children}
        </SvgLayer>
      );
    }
    return <DivLayer>{this.props.children}</DivLayer>;
  }
}

const Canvas = styled.div`
  position: relative;
  cursor: move;
  overflow: hidden;
`;

class MyCanvasWidget extends CanvasWidget {
  render() {
    const engine = this.props.engine;
    const model = engine.getModel();

    return (
      <Canvas
        className={this.props.className}
        ref={this.ref}
        onWheel={(event) => {
          this.props.engine.getActionEventBus().fireAction({ event });
        }}
        onMouseDown={(event) => {
          this.props.engine.getActionEventBus().fireAction({ event });
        }}
        onMouseUp={(event) => {
          this.props.engine.getActionEventBus().fireAction({ event });
        }}
        onMouseMove={(event) => {
          this.props.engine.getActionEventBus().fireAction({ event });
        }}
        onTouchStart={(event) => {
          this.props.engine.getActionEventBus().fireAction({ event });
        }}
        onTouchEnd={(event) => {
          this.props.engine.getActionEventBus().fireAction({ event });
        }}
        onTouchMove={(event) => {
          this.props.engine.getActionEventBus().fireAction({ event });
        }}
      >
        {model.getLayers().map((layer) => {
          return (
            <BackgroundLayerWidget layer={layer} key={layer.getID()}>
              <TransformLayerWidget layer={layer} key={layer.getID()}>
                <SmartLayerWidget
                  layer={layer}
                  engine={this.props.engine}
                  key={layer.getID()}
                />
              </TransformLayerWidget>
            </BackgroundLayerWidget>
          );
        })}
      </Canvas>
    );
  }
}

const FlowCanvas = () => {
  return (
    <CanvasWrapper onDrop={domain.onDrop} onDragOver={domain.onDragOver}>
      <MyCanvasWidget
        engine={domain.diagramEngine}
        className={"canvas-widget"}
      />
    </CanvasWrapper>
  );
};
export default FlowCanvas;
