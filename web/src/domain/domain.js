export const parseFlowToGraph = (flow) => {
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
