import { EuiFlexGroup, EuiPanel, htmlIdGenerator } from "@elastic/eui";
import React, { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";

import { listNodeTasks } from "../../../api/thunk";
import GalleryItemWidget from "../../components/GalleryItemWidget";

const NodeGallery = () => {
  const nodeTasks = useSelector((state) => state.flowEditor.nodeTasks);
  const dispatch = useDispatch();
  if (nodeTasks == null) dispatch(listNodeTasks());
  useEffect(() => {
    if (nodeTasks == null) dispatch(listNodeTasks());
  }, [nodeTasks]);
  const itemWidgets = (nodeTasks ?? []).map((task) => (
    <GalleryItemWidget
      key={htmlIdGenerator("gallery-widget")()}
      model={{ taskName: task.task_name, inputs: task.arg_names }}
      name={task.task_name}
    />
  ));
  return (
    <EuiPanel>
      <EuiFlexGroup direction="column" alignitems="center">
        {itemWidgets}
      </EuiFlexGroup>
    </EuiPanel>
  );
};

export default NodeGallery;
