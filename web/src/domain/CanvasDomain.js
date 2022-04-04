import {
  EuiButtonIcon,
  EuiFieldText,
  EuiFlexGroup,
  EuiFlexItem,
  EuiText,
  htmlIdGenerator,
} from "@elastic/eui";
import { AbstractReactFactory } from "@projectstorm/react-canvas-core";
import { DefaultNodeModel, PortWidget } from "@projectstorm/react-diagrams";
import * as SRD from "@projectstorm/react-diagrams";
import * as _ from "lodash";
import React, { useState } from "react";

import FlowNodeModel from "./FlowNodeModel";

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

class CanvasDomain {
  constructor() {
    const diagramEngine = SRD.default();
    diagramEngine.setModel(new SRD.DiagramModel());
    diagramEngine.getNodeFactories().registerFactory(new FlowNodeFactory());
    this.diagramEngine = diagramEngine;
  }
  parseFlowToGraph = (flow) => {
    const els = [];
    if (flow?.nodes == null) return;
    flow?.nodes.forEach((node) => {
      els.push({
        id: node.name,
        type: node.end_of_flow ? "output" : node.switchon ? "input" : "default",
        data: { label: node.name },
        position: {
          x: Math.random() * window.innerWidth - 100,
          y: Math.random() * window.innerHeight,
        },
      });
      node.receivers.forEach((recv) => {
        els.push({
          id: `${node.name}-send-${recv}`,
          source: node.name,
          target: recv,
          type: "smoothstep",
          animated: true,
        });
      });
      node.callees.forEach((_callee) => {
        const callee = _callee === ":self:" ? node.name : _callee;
        if (callee !== "SWITCHON") {
          els.push({
            id: `${node.name}-call-${callee}`,
            source: node.name,
            target: callee,
            type: "smoothstep",
            animated: true,
          });
        }
      });
    });
    return els;
  };

  addInPortToSelected = (name) => {
    let model = this.diagramEngine.getModel();

    _.forEach(model.getSelectedEntities(), (node) => {
      if (node instanceof DefaultNodeModel) {
        const portName = name ?? `in-${node.getInPorts().length + 1}`;
        node.addInPort(portName);
      }
    });
    this.diagramEngine.repaintCanvas();
  };
  addOutPortToSelected = (name) => {
    let model = this.diagramEngine.getModel();

    _.forEach(model.getSelectedEntities(), (node) => {
      if (node instanceof DefaultNodeModel) {
        const portName = name ?? `out-${node.getOutPorts().length + 1}`;
        node.addOutPort(portName);
      }
    });
    this.diagramEngine.repaintCanvas();
  };

  deleteInPortFromSelected = (portName) => {
    let model = this.diagramEngine.getModel();
    _.forEach(model.getSelectedEntities(), (node) => {
      const removedPorts = [];
      if (node instanceof DefaultNodeModel) {
        _.forEach(node.getInPorts(), (port) => {
          if (port.options.label === portName) {
            removedPorts.push(port);
          }
        });
        _.forEach(removedPorts, (port) => {
          node.removePort(port);
        });
      }
    });
    this.diagramEngine.repaintCanvas();
  };

  deleteOutPortFromSelected = (portName) => {
    let model = this.diagramEngine.getModel();
    _.forEach(model.getSelectedEntities(), (node) => {
      const removedPorts = [];
      if (node instanceof DefaultNodeModel) {
        _.forEach(node.getOutPorts(), (port) => {
          if (port.options.label === portName) {
            removedPorts.push(port);
          }
        });
        _.forEach(removedPorts, (port) => {
          node.removePort(port);
        });
      }
    });
    this.diagramEngine.repaintCanvas();
  };

  reroute = () => {
    const factory = this.diagramEngine
      .getLinkFactories()
      .getFactory(SRD.PathFindingLinkFactory.NAME);
    factory.calculateRoutingMatrix();
  };

  autoDistribute = (rankdir) => {
    const dagreEngine = new SRD.DagreEngine({
      graph: {
        rankdir: rankdir,
        ranker: "longest-path",
        marginx: 25,
        marginy: 25,
      },
      includeLinks: true,
    });
    dagreEngine.redistribute(this.diagramEngine.getModel());
    this.reroute(this.diagramEngine);
    this.diagramEngine.repaintCanvas();
  };

  onDrop = (event) => {
    event.preventDefault();

    const data = JSON.parse(event.dataTransfer.getData("storm-diagram-node"));

    const node = new FlowNodeModel({
      name: data.taskName,
      tasks: [data.taskName],
      inPorts: data.inputs,
      color: "rgb(192,255,0)",
    });
    node.addInPort("In");
    const point = this.diagramEngine.getRelativeMousePoint(event);
    node.setPosition(point);
    this.diagramEngine.getModel().addNode(node);
    this.diagramEngine.repaintCanvas();
  };

  onDragOver = (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  };
}

export default new CanvasDomain();
