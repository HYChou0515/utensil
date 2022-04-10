import {
  EuiAccordion,
  EuiFieldText,
  EuiFlexGroup,
  EuiFlexItem,
  EuiText,
  htmlIdGenerator,
} from "@elastic/eui";
import { AbstractReactFactory } from "@projectstorm/react-canvas-core";
import { PortWidget } from "@projectstorm/react-diagrams";
import * as _ from "lodash";
import React, { useState } from "react";

import FlowNodeModel from "../FlowNodeModel";

const FlowNodeNameWidget = ({ name }) => {
  return (
    <EuiFlexItem className="node-title-box">
      <EuiText>{name}</EuiText>
    </EuiFlexItem>
  );
};

const FlowNodeParamWidget = ({ param, value, setValue }) => {
  const [isMod, setIsMod] = useState(false);
  const [shownValue, setShownValue] = useState(value);
  const [tmpVal, setTmpVal] = useState(shownValue ?? "");
  return (
    <EuiFlexItem
      style={{ marginLeft: "3px", marginRight: "3px" }}
      onDoubleClick={() => setIsMod(true)}
    >
      {isMod ? (
        <EuiFlexGroup gutterSize={"none"} alignitems={"center"}>
          <EuiFlexItem grow={false}>
            <EuiText size={"xs"}>{`${param}=`}</EuiText>
          </EuiFlexItem>
          <EuiFlexItem style={{ fontSize: "12px", height: "16px" }}>
            <EuiFieldText
              style={{ fontSize: "12px", height: "16px" }}
              autoFocus
              onBlur={() => {
                setIsMod(false);
                setShownValue(tmpVal);
                setValue(tmpVal);
              }}
              value={tmpVal}
              onChange={(e) => setTmpVal(e.target.value)}
            />
          </EuiFlexItem>
        </EuiFlexGroup>
      ) : (
        <EuiText size={"xs"}>{`${param}=${shownValue}`}</EuiText>
      )}
    </EuiFlexItem>
  );
};

class FlowNodeWidget extends React.Component {
  render() {
    const inPorts = this.props.node.inPorts.map((p) => (
      <EuiFlexItem
        className="left-port"
        key={htmlIdGenerator("left-port")()}
        grow={false}
      >
        <EuiFlexGroup gutterSize={"none"}>
          <EuiFlexItem grow={false}>
            <PortWidget
              engine={this.props.engine}
              port={this.props.node.getPort(p)}
            >
              <div className="circle-port" />
            </PortWidget>
          </EuiFlexItem>
          <EuiFlexItem>
            <div>
              <h3>{p}</h3>
            </div>
          </EuiFlexItem>
        </EuiFlexGroup>
      </EuiFlexItem>
    ));
    inPorts.push(
      <EuiFlexItem
        className="left-port"
        key={htmlIdGenerator("left-port")()}
        grow={false}
      >
        <EuiFlexGroup gutterSize={"none"}>
          <EuiFlexItem grow={false}>
            <PortWidget
              engine={this.props.engine}
              port={this.props.node.getPort("trigger")}
            >
              <div className="trigger-port" />
            </PortWidget>
          </EuiFlexItem>
        </EuiFlexGroup>
      </EuiFlexItem>
    );
    const outPorts = this.props.node.outPorts.map((p) => (
      <EuiFlexItem
        className="right-port"
        key={htmlIdGenerator("right-port")()}
        grow={false}
      >
        <EuiFlexGroup gutterSize={"none"}>
          <EuiFlexItem>
            <div>
              <h3>{p}</h3>
            </div>
          </EuiFlexItem>
          <EuiFlexItem>
            <PortWidget
              engine={this.props.engine}
              port={this.props.node.getPort(p)}
            >
              <div className="circle-port" />
            </PortWidget>
          </EuiFlexItem>
        </EuiFlexGroup>
      </EuiFlexItem>
    ));
    const requiredParams = _.zip(
      this.props.node.params,
      this.props.node.paramValues
    )
      .filter(([[par, req], v]) => req === "required")
      .map(([[par, req], v], index) => (
        <FlowNodeParamWidget
          key={`required-param-${index}`}
          param={par}
          value={v}
          setValue={(newV) => {
            this.props.node.paramValues[index] = newV;
          }}
        />
      ));
    const requiredAllFulfilled =
      _.zip(this.props.node.params, this.props.node.paramValues).filter(
        ([[par, req], v]) => req === "required" && !v
      ).length === 0;
    const optionalParams = _.zip(
      this.props.node.params,
      this.props.node.paramValues
    )
      .filter(([[par, req], v]) => req === "optional")
      .map(([[par, req], v], index) => (
        <FlowNodeParamWidget
          key={`optional-param-${index}`}
          param={par}
          value={v}
          setValue={(newV) => {
            this.props.node.paramValues[index] = newV;
          }}
        />
      ));
    return (
      <EuiFlexGroup
        direction="column"
        gutterSize={"none"}
        className={`custom-node ${
          this.props.node.isSelected() ? "selected-custom-node" : ""
        }`}
        style={{ backgroundColor: this.props.node.color }}
        justifyContent="spaceAround"
      >
        <FlowNodeNameWidget name={this.props.node.task} />

        <EuiFlexItem>
          <EuiFlexGroup gutterSize={"none"}>
            <EuiFlexItem>
              <EuiFlexGroup
                direction="column"
                gutterSize={"none"}
                className="left-port-column"
              >
                {inPorts}
              </EuiFlexGroup>
            </EuiFlexItem>

            <EuiFlexItem
              style={{
                marginLeft: "5px",
                marginRight: "5px",
                marginBottom: "5px",
              }}
            >
              {requiredParams.length > 0 && (
                <EuiAccordion
                  initialIsOpen={true}
                  id={htmlIdGenerator("accordion")()}
                  buttonContent={
                    <EuiText size={"xs"} color={!requiredAllFulfilled && "red"}>
                      required
                    </EuiText>
                  }
                >
                  <EuiFlexGroup direction="column" gutterSize={"none"}>
                    {requiredParams}
                  </EuiFlexGroup>
                </EuiAccordion>
              )}
              {optionalParams.length > 0 && (
                <EuiAccordion
                  id={htmlIdGenerator("accordion")()}
                  buttonContent="optional"
                >
                  <EuiFlexGroup direction="column" gutterSize={"none"}>
                    {optionalParams}
                  </EuiFlexGroup>
                </EuiAccordion>
              )}
            </EuiFlexItem>

            <EuiFlexItem>
              <EuiFlexGroup
                direction="column"
                gutterSize={"none"}
                className="right-port-column"
              >
                {outPorts}
              </EuiFlexGroup>
            </EuiFlexItem>
          </EuiFlexGroup>
        </EuiFlexItem>
      </EuiFlexGroup>
    );
  }
}

class FlowNodeFactory extends AbstractReactFactory {
  constructor() {
    super("flow-node");
  }

  generateModel(event) {
    return new FlowNodeModel();
  }

  generateReactWidget(event) {
    return <FlowNodeWidget engine={this.engine} node={event.model} />;
  }
}

export default FlowNodeFactory;
