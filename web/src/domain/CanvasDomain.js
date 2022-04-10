import * as SRD from "@projectstorm/react-diagrams";
import * as _ from "lodash";
import React from "react";

import FlowNodeFactory from "./components/FlowNodeFactory";
import FlowNodeModel from "./FlowNodeModel";
import { MyZoomCanvasAction } from "./MyZoomCanvasAction";

class CanvasDomain {
  constructor() {
    const diagramEngine = SRD.default({
      registerDefaultZoomCanvasAction: false,
    });
    diagramEngine.eventBus.registerAction(new MyZoomCanvasAction());
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
      if (node instanceof SRD.DefaultNodeModel) {
        const portName = name ?? `in-${node.getInPorts().length + 1}`;
        node.addInPort(portName);
      }
    });
    this.diagramEngine.repaintCanvas();
  };
  addOutPortToSelected = (name) => {
    let model = this.diagramEngine.getModel();

    _.forEach(model.getSelectedEntities(), (node) => {
      if (node instanceof SRD.DefaultNodeModel) {
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
      if (node instanceof SRD.DefaultNodeModel) {
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
      if (node instanceof SRD.DefaultNodeModel) {
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

    const data = JSON.parse(event.dataTransfer.getData("dnd-flow-node"));

    let nodeOption;
    switch (data.type) {
      case "task":
        nodeOption = {
          nodeType: data.type,
          name: data.name,
          inPorts: data.inputs,
          params: data.params,
          module: data.module,
          color: data.color,
        };
        break;
      case "switch-on":
        nodeOption = {
          nodeType: data.type,
          name: data.name,
          inPorts: [],
          params: [],
          color: data.color,
          hasTrigger: false,
        };
        break;
      case "end-of-flow":
        nodeOption = {
          nodeType: data.type,
          name: data.name,
          inPorts: [],
          params: [],
          color: data.color,
          outPorts: [],
        };
        break;
    }
    const node = new FlowNodeModel(nodeOption);
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
