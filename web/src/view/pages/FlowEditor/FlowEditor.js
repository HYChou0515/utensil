import {
  EuiFlexGroup,
  EuiFlexItem,
  EuiPageTemplate,
  EuiResizableContainer,
} from "@elastic/eui";
import React from "react";
import { useDispatch, useSelector } from "react-redux";

import { createCollapseRightSideFunc } from "../../../store/features/canvas/flowEditor";
import EditorTools from "./EditorTools";
import FlowCanvas from "./FlowCanvas";
import Menu from "./Menu";
import NodeGallery from "./NodeGallery";
import OpenFileUi from "./OpenFileUi";
import SettingUi from "./SettingUi";

const FlowEditor = () => {
  const isShowOpenFileUi = useSelector(
    (state) => state.flowEditor.isShowOpenFileUi
  );
  const isShowSettingUi = useSelector(
    (state) => state.flowEditor.isShowSettingUi
  );
  const dispatch = useDispatch();
  return (
    <>
      {isShowOpenFileUi && <OpenFileUi />}
      {isShowSettingUi && <SettingUi />}
      <EuiPageTemplate pageContentProps={{ paddingSize: "none" }}>
        <Menu />
        <EuiResizableContainer>
          {(EuiResizablePanel, EuiResizableButton, { togglePanel }) => {
            dispatch(
              createCollapseRightSideFunc((id, direction) =>
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
                      <EditorTools />
                    </EuiFlexItem>
                    <EuiFlexItem>
                      <NodeGallery />
                    </EuiFlexItem>
                  </EuiFlexGroup>
                </EuiResizablePanel>
              </>
            );
          }}
        </EuiResizableContainer>
      </EuiPageTemplate>
    </>
  );
};

export default FlowEditor;
