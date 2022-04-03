import {
  EuiButton,
  EuiModal,
  EuiModalBody,
  EuiModalFooter,
  EuiModalHeader,
  EuiModalHeaderTitle,
} from "@elastic/eui";
import { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";

import { listNodeTasks } from "../../../api/thunk";
import { closeSettingUi } from "../../../store/features/canvas/flowEditor";
import DataGrid from "../../components/DataGrid";

const SettingUi = () => {
  const dispatch = useDispatch();
  const nodeTasks = useSelector((state) => state.flowEditor.nodeTasks);

  useEffect(() => {
    dispatch(listNodeTasks());
  }, []);

  const columns = [
    { id: "key", actions: false },
    { id: "module", initialWidth: 400, actions: false },
    { id: "task_name", actions: false },
  ];
  return (
    <EuiModal
      style={{ minWidth: 1200 }}
      onClose={() => dispatch(closeSettingUi())}
    >
      <EuiModalHeader>
        <EuiModalHeaderTitle>
          <h1>title</h1>
        </EuiModalHeaderTitle>
      </EuiModalHeader>

      <EuiModalBody>
        <DataGrid data={nodeTasks} columns={columns} />
      </EuiModalBody>

      <EuiModalFooter>
        <EuiButton onClick={() => dispatch(closeSettingUi())} fill>
          Close
        </EuiButton>
      </EuiModalFooter>
    </EuiModal>
  );
};

export default SettingUi;
