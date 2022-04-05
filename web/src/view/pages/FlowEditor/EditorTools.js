import { EuiFlexGrid, EuiFlexItem, EuiPanel } from "@elastic/eui";
import React from "react";
import { useDispatch } from "react-redux";

import {
  addInPortToSelected,
  addOutPortToSelected,
  deleteInPortFromSelected,
  deleteOutPortFromSelected,
} from "../../../store/features/canvas/flowEditor";
import {
  AddInPortIcon,
  AddOutPortIcon,
  DeleteInPortIcon,
  DeleteOutPortIcon,
} from "../../components/Icons";
import TextPopover from "../../components/TextPopover";

const EditorTools = () => {
  const dispatch = useDispatch();
  return (
    <EuiPanel>
      <EuiFlexGrid direction="column" alignitems="center">
        <EuiFlexItem>
          <TextPopover
            iconType={AddInPortIcon}
            display="base"
            placeholder="name of the in port ..."
            onSubmit={(name) => dispatch(addInPortToSelected(name))}
          />
        </EuiFlexItem>
        <EuiFlexItem>
          <TextPopover
            iconType={AddOutPortIcon}
            display="base"
            placeholder="name of the out port ..."
            onSubmit={(name) => dispatch(addOutPortToSelected(name))}
          />
        </EuiFlexItem>
        <EuiFlexItem>
          <TextPopover
            iconType={DeleteInPortIcon}
            display="base"
            placeholder="name of the in port ..."
            onSubmit={(name) => dispatch(deleteInPortFromSelected(name))}
          />
        </EuiFlexItem>
        <EuiFlexItem>
          <TextPopover
            iconType={DeleteOutPortIcon}
            display="base"
            placeholder="name of the out port ..."
            onSubmit={(name) => dispatch(deleteOutPortFromSelected(name))}
          />
        </EuiFlexItem>
      </EuiFlexGrid>
    </EuiPanel>
  );
};

export default EditorTools;
