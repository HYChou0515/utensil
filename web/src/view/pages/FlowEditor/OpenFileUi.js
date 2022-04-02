import {
  EuiButton,
  EuiButtonEmpty,
  EuiFilePicker,
  EuiForm,
  EuiModal,
  EuiModalBody,
  EuiModalFooter,
  EuiModalHeader,
  EuiModalHeaderTitle,
} from "@elastic/eui";
import React, { useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { v4 } from "uuid";

import { uploadFlow } from "../../../store/actions";
import {
  closeOpenFileUi,
  setLoading,
} from "../../../store/features/canvas/flowEditor";

const OpenFileUi = () => {
  const dispatch = useDispatch();
  const isLoading = useSelector((state) => state.flowEditor.isLoading);

  const formId = v4();
  const [file, setFile] = useState();
  const [isInvalidFile, setIsInvalidFile] = useState(false);
  const isLoadingUpload = isLoading === "upload file";
  const onOpenClick = () => {
    if (file) {
      setIsInvalidFile(false);
      dispatch(setLoading("upload file"));
      const formData = new FormData();
      formData.append("file", file, file.name);
      dispatch({ type: uploadFlow.type, payload: formData });
    } else {
      setIsInvalidFile(true);
    }
  };
  return (
    <EuiModal onClose={() => dispatch(closeOpenFileUi())}>
      <EuiModalHeader>
        <EuiModalHeaderTitle>
          <h1>Drop a file here or click to select a file</h1>
        </EuiModalHeaderTitle>
      </EuiModalHeader>

      <EuiModalBody>
        <EuiForm id={formId}>
          <EuiFilePicker
            onChange={(f) => setFile(f.item(0))}
            isLoading={isLoadingUpload}
            isInvalid={isInvalidFile}
          />
        </EuiForm>
      </EuiModalBody>

      <EuiModalFooter>
        <EuiButtonEmpty onClick={() => dispatch(closeOpenFileUi())}>
          Cancel
        </EuiButtonEmpty>
        <EuiButton
          type="submit"
          form={formId}
          fill
          onClick={onOpenClick}
          isDisabled={(file?.size ?? 0) === 0}
          isLoading={isLoadingUpload}
        >
          Open
        </EuiButton>
      </EuiModalFooter>
    </EuiModal>
  );
};

export default OpenFileUi;
