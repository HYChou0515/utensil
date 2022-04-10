import { EuiFlexGroup, EuiPanel, htmlIdGenerator } from "@elastic/eui";
import React, { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";

import { listNodeTasks } from "../../../api/thunk";
import { strToColor } from "../../../domain/misc";
import GalleryItemWidget from "../../components/GalleryItemWidget";

const NodeGallery = () => {
  const nodeTasks = useSelector((state) => state.flowEditor.nodeTasks);
  const dispatch = useDispatch();
  if (nodeTasks == null) dispatch(listNodeTasks());
  useEffect(() => {
    if (nodeTasks == null) dispatch(listNodeTasks());
  }, [nodeTasks]);
  const staticItemWidgets = [
    <GalleryItemWidget
      key={htmlIdGenerator("gallery-widget")()}
      model={{
        type: "switch-on",
        color: "#ff471a",
        name: "Switch On",
      }}
    />,
    <GalleryItemWidget
      key={htmlIdGenerator("gallery-widget")()}
      model={{
        type: "end-of-flow",
        color: "#ff471a",
        name: "End of Flow",
      }}
    />,
  ];
  const taskItemWidgets = (nodeTasks ?? []).map((task) => (
    <GalleryItemWidget
      key={htmlIdGenerator("gallery-widget")()}
      model={{
        type: "task",
        color: strToColor(task.module),
        name: task.task_name,
        inputs: task.arg_names,
        params: task.params,
      }}
    />
  ));
  return (
    <EuiPanel>
      <EuiFlexGroup direction="column" alignitems="center">
        {[...staticItemWidgets, taskItemWidgets]}
      </EuiFlexGroup>
    </EuiPanel>
  );
};

export default NodeGallery;
