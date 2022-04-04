import {
  EuiButtonIcon,
  EuiFieldText,
  EuiFlexGroup,
  EuiFlexItem,
  EuiText,
  htmlIdGenerator,
} from "@elastic/eui";
import { AbstractReactFactory } from "@projectstorm/react-canvas-core";
import { PortWidget } from "@projectstorm/react-diagrams";
import React, { useState } from "react";

import FlowNodeModel from "../FlowNodeModel";

const FlowNodeNameWidget = ({ initName, onSetName }) => {
  const [name, setName] = useState(initName);
  const [formTmpName, setFormTmpName] = useState(name);
  const [isChangeName, setIsChangeName] = useState(false);
  const onConfirm = () => {
    setIsChangeName(false);
    setName(formTmpName);
    onSetName(formTmpName);
  };
  const onCancel = () => {
    setIsChangeName(false);
    setFormTmpName(name);
  };
  return (
    <EuiFlexItem
      className="node-title-box"
      onDoubleClick={() => setIsChangeName(true)}
    >
      {isChangeName ? (
        <EuiFlexGroup gutterSize={"xs"} alignItems="center">
          <EuiFlexItem>
            <EuiFieldText
              value={formTmpName}
              onChange={(e) => setFormTmpName(e.target.value)}
            />
          </EuiFlexItem>
          <EuiFlexItem grow={false}>
            <EuiButtonIcon
              iconType={"check"}
              color={"success"}
              onClick={onConfirm}
              display="fill"
            />
          </EuiFlexItem>
          <EuiFlexItem grow={false}>
            <EuiButtonIcon
              iconType={"cross"}
              color={"danger"}
              onClick={onCancel}
              display="fill"
            />
          </EuiFlexItem>
        </EuiFlexGroup>
      ) : (
        <EuiText>{name}</EuiText>
      )}
    </EuiFlexItem>
  );
};

class FlowNodeWidget extends React.Component {
  render() {
    const inPorts = this.props.node.inPorts.map((p) => (
      <EuiFlexItem className="left-port" key={htmlIdGenerator("left-port")()}>
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
      <EuiFlexItem className="left-port" key={htmlIdGenerator("left-port")()}>
        <EuiFlexGroup gutterSize={"none"}>
          <EuiFlexItem>
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
    const tasks = this.props.node.tasks.map((p) => (
      <EuiFlexItem className="task-box" key={htmlIdGenerator("task-box")()}>
        <h3>{p}</h3>
      </EuiFlexItem>
    ));
    return (
      <div className="custom-node">
        <EuiFlexGroup
          direction="column"
          gutterSize={"none"}
          justifyContent="spaceAround"
        >
          <FlowNodeNameWidget
            initName={this.props.node.name}
            onSetName={(newName) => {
              this.props.node.name = newName;
            }}
          />

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

              <EuiFlexItem>
                <EuiFlexGroup direction="column" className="task-box-column">
                  {tasks}
                </EuiFlexGroup>
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
      </div>
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
