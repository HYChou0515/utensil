import {
  EuiButton,
  EuiButtonEmpty,
  EuiFilePicker,
  EuiFlexGrid,
  EuiFlexGroup,
  EuiFlexItem,
  EuiForm,
  EuiModal,
  EuiModalBody,
  EuiModalFooter,
  EuiModalHeader,
  EuiModalHeaderTitle,
  EuiPageTemplate,
  EuiPanel,
  EuiResizableContainer,
} from "@elastic/eui";
import { CanvasWidget } from "@projectstorm/react-canvas-core";
import React, { useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { v4 } from "uuid";

import domain from "../../../domain/CanvasDomain";
import { uploadFlow } from "../../../store/actions";
import {
  addInPortToSelected,
  addOutPortToSelected,
  closeOpenFileUi,
  createCollapseFn,
  deleteInPortFromSelected,
  deleteOutPortFromSelected,
  setLoading,
} from "../../../store/features/canvas/flowEditor";
import CanvasWrapper from "../../components/CanvasWrapper";
import GalleryItemWidget from "../../components/GalleryItemWidget";
import {
  AddInPortIcon,
  AddOutPortIcon,
  DeleteInPortIcon,
  DeleteOutPortIcon,
} from "../../components/Icons";
import TextPopover from "../../components/TextPopover";
import Menu from "./Menu";

const NodeGallery = () => {
  return (
    <EuiPanel>
      <EuiFlexGroup direction="column" alignitems="center">
        <EuiFlexItem>
          <GalleryItemWidget
            model={{ type: "in", color: "rgb(192,255,0)" }}
            name="Input Node"
          />
        </EuiFlexItem>
        <EuiFlexItem>
          <GalleryItemWidget
            model={{ type: "out", color: "rgb(0,192,255)" }}
            name="Output Node"
          />
        </EuiFlexItem>
      </EuiFlexGroup>
    </EuiPanel>
  );
};

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

const FlowCanvas = () => {
  return (
    <CanvasWrapper onDrop={domain.onDrop} onDragOver={domain.onDragOver}>
      <CanvasWidget engine={domain.diagramEngine} className={"canvas-widget"} />
    </CanvasWrapper>
  );
};

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

const Editor = () => {
  const isShowOpenFileUi = useSelector(
    (state) => state.flowEditor.isShowOpenFileUi
  );
  const dispatch = useDispatch();
  return (
    <>
      {isShowOpenFileUi && <OpenFileUi />}
      <EuiPageTemplate pageContentProps={{ paddingSize: "none" }}>
        <Menu />
        <EuiFlexGroup>
          <EuiFlexItem>
            <EuiResizableContainer>
              {(EuiResizablePanel, EuiResizableButton, { togglePanel }) => {
                dispatch(
                  createCollapseFn((id, direction) =>
                    togglePanel(id, { direction })
                  )
                );
                return (
                  <>
                    <EuiResizablePanel
                      id="panel-canvas"
                      initialSize={80}
                      minSize="50%"
                    >
                      <FlowCanvas />
                    </EuiResizablePanel>

                    <EuiResizableButton />

                    <EuiResizablePanel
                      id="panel-gallery"
                      initialSize={20}
                      minSize={`${200}px`}
                    >
                      <EuiFlexGroup direction="column" alignitems="center">
                        <EuiFlexItem>
                          <NodeGallery />
                        </EuiFlexItem>
                        <EuiFlexItem>
                          <EditorTools />
                        </EuiFlexItem>
                      </EuiFlexGroup>
                    </EuiResizablePanel>
                  </>
                );
              }}
            </EuiResizableContainer>
          </EuiFlexItem>
        </EuiFlexGroup>
      </EuiPageTemplate>
    </>
  );
};

const FlowEditor = () => {
  return <Editor />;
};

export default FlowEditor;
