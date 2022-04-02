import { DefaultNodeModel } from "@projectstorm/react-diagrams";
import * as SRD from "@projectstorm/react-diagrams";
import * as _ from "lodash";

class CanvasDomain {
  constructor() {
    const diagramEngine = SRD.default();
    diagramEngine.setModel(new SRD.DiagramModel());
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
    const nodesCount = _.keys(this.diagramEngine.getModel().getNodes()).length;

    let node = null;
    if (data.type === "in") {
      node = new SRD.DefaultNodeModel(
        "Node " + (nodesCount + 1),
        "rgb(192,255,0)"
      );
      node.addInPort("In");
    } else {
      node = new SRD.DefaultNodeModel(
        "Node " + (nodesCount + 1),
        "rgb(0,192,255)"
      );
      node.addOutPort("Out");
    }
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
