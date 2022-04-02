import {
  EuiFlexGroup,
  EuiFlexItem,
  EuiPageTemplate,
  EuiResizableContainer,
} from "@elastic/eui";
import React from "react";
import { useDispatch, useSelector } from "react-redux";

import { createCollapseFn } from "../../../store/features/canvas/flowEditor";
import EditorTools from "./EditorTools";
import FlowCanvas from "./FlowCanvas";
import Menu from "./Menu";
import NodeGallery from "./NodeGallery";
import OpenFileUi from "./OpenFileUi";

const FlowEditor = () => {
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

export default FlowEditor;
