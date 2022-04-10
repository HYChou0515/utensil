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
    this.task = options.task;
    this.color = options.color;
    this.hasTrigger = options.hasTrigger ?? true;
    this.inPorts = options.inPorts || [];
    this.outPorts = options.outPorts || ["out"];
    this.params = options.params || [];
    this.paramValues = options.params.map(() => null);
    _.forEach(this.inPorts, (p) =>
      this.addPort(
        new DefaultPortModel({
          in: true,
          name: p,
        })
      )
    );
    if (this.hasTrigger) {
      this.addPort(
        new TriggerPortModel({
          in: true,
          name: "trigger",
        })
      );
    }
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
      color: this.color,
      task: this.task,
      inPorts: this.inPorts,
      outPorts: this.outPorts,
      params: this.params,
      paramValues: this.paramValues,
    };
  }

  deserialize(ob, engine) {
    super.deserialize(ob, engine);
    this.tasks = ob.tasks;
  }
}

export default FlowNodeModel;
