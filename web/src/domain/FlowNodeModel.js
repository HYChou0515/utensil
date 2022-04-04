import {
  DefaultNodeModel,
  DefaultPortModel,
} from "@projectstorm/react-diagrams";
import * as _ from "lodash";
import React from "react";

class TriggerPortModel extends DefaultPortModel {
  constructor(options) {
    super(options);
    this.type = "trigger";
  }
}

class FlowNodeModel extends DefaultNodeModel {
  constructor(options = {}) {
    super({
      ...options,
      type: "flow-node",
    });
    this.tasks = options.tasks || [];
    this.color = options.color || { options: "red" };
    this.name = options.name;
    this.inPorts = options.inPorts || [];
    this.outPorts = options.outPorts || ["out"];
    _.forEach(this.inPorts, (p) =>
      this.addPort(
        new DefaultPortModel({
          in: true,
          name: p,
        })
      )
    );
    this.addPort(
      new TriggerPortModel({
        in: true,
        name: "trigger",
      })
    );
    _.forEach(this.outPorts, (p) =>
      this.addPort(
        new DefaultPortModel({
          in: false,
          name: p,
        })
      )
    );
  }

  serialize() {
    return {
      ...super.serialize(),
      name: this.name,
      color: this.color,
      tasks: this.tasks,
      inPorts: this.inPorts,
      outPorts: this.outPorts,
    };
  }

  deserialize(ob, engine) {
    super.deserialize(ob, engine);
    this.tasks = ob.tasks;
  }
}

export default FlowNodeModel;
