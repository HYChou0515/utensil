import {
  EuiButton,
  EuiButtonIcon,
  EuiCodeBlock,
  EuiEmptyPrompt,
  EuiLoadingSpinner,
  EuiModal,
  EuiModalBody,
  EuiModalFooter,
  EuiModalHeader,
  EuiModalHeaderTitle,
  EuiText,
} from "@elastic/eui";
import React, { Fragment, useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";

import apiClient from "../../../api/api";
import { listNodeTasks } from "../../../api/thunk";
import { closeSettingUi } from "../../../store/features/canvas/flowEditor";
import DataGrid from "../../components/DataGrid";

const SourceCodeViewCell = (item) => {
  let modal = null;
  const [showModal, _setShowModal] = useState(false);
  const [loading, setIsLoading] = useState(false);
  const [sourceCode, setSourceCode] = useState(null);

  const setShowModal = (open) => {
    if (open && sourceCode == null) {
      setIsLoading(true);
      apiClient.getSourceCode(item.module, item.task_name).then((code) => {
        setSourceCode(code);
        setIsLoading(false);
      });
    }
    _setShowModal(open);
  };
  if (showModal) {
    modal = (
      <EuiModal onClose={() => setShowModal(false)}>
        <EuiModalHeader>
          <EuiModalHeaderTitle>
            <h1>Source Code</h1>
          </EuiModalHeaderTitle>
        </EuiModalHeader>

        <EuiModalBody>
          <EuiText>{`${item.module}.${item.task_name}`}</EuiText>
          {loading ? (
            <EuiEmptyPrompt
              icon={<EuiLoadingSpinner size="xl" />}
              title={<h2>Loading</h2>}
            />
          ) : (
            <EuiCodeBlock overflowHeight={400} language="python" isCopyable>
              {sourceCode}
            </EuiCodeBlock>
          )}
        </EuiModalBody>

        <EuiModalFooter>
          <EuiButton onClick={() => setShowModal(false)} fill>
            Close
          </EuiButton>
        </EuiModalFooter>
      </EuiModal>
    );
  }

  return (
    <Fragment>
      <EuiButtonIcon
        color="text"
        iconType="eye"
        iconSize="s"
        aria-label="View details"
        onClick={() => setShowModal(!showModal)}
      />
      {modal}
    </Fragment>
  );
};

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
          <h1>Node Tasks</h1>
        </EuiModalHeaderTitle>
      </EuiModalHeader>

      <EuiModalBody>
        <DataGrid
          data={nodeTasks}
          columns={columns}
          leadingControlColumns={[
            {
              id: "View",
              width: 36,
              headerCellRender: () => null,
              rowCellRender: SourceCodeViewCell,
            },
          ]}
        />
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
