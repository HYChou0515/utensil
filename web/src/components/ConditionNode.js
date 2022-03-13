import React from "react";
import { Handle } from "react-flow-renderer";

const conditionNodeStyles = {
  background: "#9CA8B3",
  color: "#FFF",
  padding: 10,
};

const ConditionNode = ({ data }) => {
  return (
    <div>
      <Handle type="target" position="left" style={{ borderRadius: 0 }} />
      <div>Hi</div>
      <Handle
        type="source"
        position="right"
        id="a"
        style={{ top: "30%", borderRadius: 0 }}
      />
      <Handle
        type="source"
        position="right"
        id="b"
        style={{ top: "70%", borderRadius: 0 }}
      />
    </div>
  );
};

export default ConditionNode;
