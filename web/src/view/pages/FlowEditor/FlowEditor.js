import {
  EuiButton,
  EuiButtonEmpty,
  EuiButtonIcon,
  EuiFilePicker,
  EuiFlexGrid,
  EuiFlexGroup,
  EuiFlexItem,
  EuiForm,
  EuiHeader,
  EuiHeaderSection,
  EuiHeaderSectionItem,
  EuiHeaderSectionItemButton,
  EuiIcon,
  EuiModal,
  EuiModalBody,
  EuiModalFooter,
  EuiModalHeader,
  EuiModalHeaderTitle,
  EuiPageTemplate,
  EuiPanel,
  EuiPopover,
  EuiPopoverFooter,
  EuiResizableContainer,
} from "@elastic/eui";
import { CanvasWidget } from "@projectstorm/react-canvas-core";
import * as SRD from "@projectstorm/react-diagrams";
import { DefaultNodeModel } from "@projectstorm/react-diagrams";
import * as _ from "lodash";
import React, {
  createContext,
  useContext,
  useEffect,
  useReducer,
  useRef,
  useState,
} from "react";
import { FaCubes, FaFolderOpen, FaSitemap } from "react-icons/fa";
import { useDispatch, useSelector } from "react-redux";
import { v4 } from "uuid";

import { getParsedFlow } from "../../../api/api";
import { parseFlowToGraph } from "../../../domain/domain";
import logo from "../../../logo.svg";
import { uploadFlow } from "../../../store/actions";
import {
  closeOpenFileUi,
  setLoading,
  toggleShowOpenFileUi,
} from "../../../store/features/canvas/flowEditor";
import CanvasWrapper from "../../components/CanvasWrapper";
import GalleryItemWidget from "../../components/GalleryItemWidget";
import { useDiagramEngine, useForceUpdate } from "../../components/hooks";
import {
  AddInPortIcon,
  AddOutPortIcon,
  DeleteInPortIcon,
  DeleteOutPortIcon,
} from "../../components/Icons";
import TextPopover from "../../components/TextPopover";
import Menu from "./Menu";

const GraphContext = createContext();
const DiagramEngineContext = createContext();
const DiagramControlContext = createContext();

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
  const { eventQueue, setNewEventComing } = useContext(DiagramControlContext);
  const onAddInPortToSelected = (name) => {
    eventQueue.push(["addInPortToSelected", { name: name }]);
    setNewEventComing(1);
  };
  const onAddOutPortToSelected = (name) => {
    eventQueue.push(["addOutPortToSelected", { name: name }]);
    setNewEventComing(1);
  };
  const onDeleteInPortFromSelected = (name) => {
    eventQueue.push(["deleteInPortFromSelected", { name: name }]);
    setNewEventComing(1);
  };
  const onDeleteOutPortFromSelected = (name) => {
    eventQueue.push(["deleteOutPortFromSelected", { name: name }]);
    setNewEventComing(1);
  };

  return (
    <EuiPanel>
      <EuiFlexGrid direction="column" alignitems="center">
        <EuiFlexItem>
          <TextPopover
            iconType={AddInPortIcon}
            display="base"
            placeholder="name of the in port ..."
            onSubmit={onAddInPortToSelected}
          />
        </EuiFlexItem>
        <EuiFlexItem>
          <TextPopover
            iconType={AddOutPortIcon}
            display="base"
            placeholder="name of the out port ..."
            onSubmit={onAddOutPortToSelected}
          />
        </EuiFlexItem>
        <EuiFlexItem>
          <TextPopover
            iconType={DeleteInPortIcon}
            display="base"
            placeholder="name of the in port ..."
            onSubmit={onDeleteInPortFromSelected}
          />
        </EuiFlexItem>
        <EuiFlexItem>
          <TextPopover
            iconType={DeleteOutPortIcon}
            display="base"
            placeholder="name of the out port ..."
            onSubmit={onDeleteOutPortFromSelected}
          />
        </EuiFlexItem>
      </EuiFlexGrid>
    </EuiPanel>
  );
};

const FlowCanvas = () => {
  const diagramEngine = useContext(DiagramEngineContext);
  const { eventQueue, newEventComing, setNewEventComing } = useContext(
    DiagramControlContext
  );

  useEffect(() => {
    while (eventQueue.length > 0) {
      const [event, args] = eventQueue.shift();
      if (event === "addInPortToSelected") {
        const { name } = args;
        addInPortToSelected(name);
      } else if (event === "addOutPortToSelected") {
        const { name } = args;
        addOutPortToSelected(name);
      } else if (event === "deleteInPortFromSelected") {
        const { name } = args;
        deleteInPortFromSelected(name);
      } else if (event === "deleteOutPortFromSelected") {
        const { name } = args;
        deleteOutPortFromSelected(name);
      } else if (event === "autoDistribute") {
        const { rankdir } = args;
        autoDistribute(rankdir);
      }
    }
    setNewEventComing(0);
  }, [newEventComing]);

  const onDragOver = (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  };

  const forceUpdate = useForceUpdate();

  const addInPortToSelected = (name) => {
    let model = diagramEngine.getModel();

    _.forEach(model.getSelectedEntities(), (node) => {
      if (node instanceof DefaultNodeModel) {
        const portName = name ?? `in-${node.getInPorts().length + 1}`;
        node.addInPort(portName);
      }
    });
    forceUpdate();
  };

  const addOutPortToSelected = (name) => {
    let model = diagramEngine.getModel();

    _.forEach(model.getSelectedEntities(), (node) => {
      if (node instanceof DefaultNodeModel) {
        const portName = name ?? `out-${node.getOutPorts().length + 1}`;
        node.addOutPort(portName);
      }
    });
    forceUpdate();
  };

  const deleteInPortFromSelected = (portName) => {
    let model = diagramEngine.getModel();
    _.forEach(model.getSelectedEntities(), (node) => {
      const removedPorts = [];
      if (node instanceof DefaultNodeModel) {
        _.forEach(node.getInPorts(), (port) => {
          if (port.options.label === portName) {
            removedPorts.push(port);
          }
        });
        _.forEach(removedPorts, (port) => {
          node.removePort(port);
        });
      }
    });
    forceUpdate();
  };

  const deleteOutPortFromSelected = (portName) => {
    let model = diagramEngine.getModel();
    _.forEach(model.getSelectedEntities(), (node) => {
      const removedPorts = [];
      if (node instanceof DefaultNodeModel) {
        _.forEach(node.getOutPorts(), (port) => {
          if (port.options.label === portName) {
            removedPorts.push(port);
          }
        });
        _.forEach(removedPorts, (port) => {
          node.removePort(port);
        });
      }
    });
    forceUpdate();
  };

  const reroute = () => {
    const factory = diagramEngine
      .getLinkFactories()
      .getFactory(SRD.PathFindingLinkFactory.NAME);
    factory.calculateRoutingMatrix();
  };

  const autoDistribute = (rankdir) => {
    const dagreEngine = new SRD.DagreEngine({
      graph: {
        rankdir: rankdir,
        ranker: "longest-path",
        marginx: 25,
        marginy: 25,
      },
      includeLinks: true,
    });
    dagreEngine.redistribute(diagramEngine.getModel());
    reroute();
    diagramEngine.repaintCanvas();
  };

  const onDrop = (event) => {
    event.preventDefault();

    const data = JSON.parse(event.dataTransfer.getData("storm-diagram-node"));
    const nodesCount = _.keys(diagramEngine.getModel().getNodes()).length;

    let node = null;
    if (data.type === "in") {
      node = new SRD.DefaultNodeModel(
        "Node " + (nodesCount + 1),
        "rgb(192,255,0)"
      );
      node.addInPort("In");
    } else {
      node = new SRD.DefaultNodeModel(
        "Node " + (nodesCount + 1),
        "rgb(0,192,255)"
      );
      node.addOutPort("Out");
    }
    const point = diagramEngine.getRelativeMousePoint(event);
    node.setPosition(point);
    diagramEngine.getModel().addNode(node);
    forceUpdate();
  };

  return (
    <CanvasWrapper onDrop={onDrop} onDragOver={onDragOver}>
      <CanvasWidget engine={diagramEngine} className={"canvas-widget"} />
    </CanvasWrapper>
  );
};

const OpenFileUi = ({ onClose }) => {
  const dispatch = useDispatch();
  const isLoading = useSelector((state) => state.isLoading);

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

  const collapseFn = useRef(() => {});
  const [graph, setGraph] = useState();
  const [flow, setFlow] = useState();
  return (
    <GraphContext.Provider value={{ graph, setGraph, flow, setFlow }}>
      {isShowOpenFileUi && <OpenFileUi />}
      <EuiPageTemplate pageContentProps={{ paddingSize: "none" }}>
        <Menu />
        <EuiFlexGroup>
          <EuiFlexItem>
            <EuiResizableContainer>
              {(EuiResizablePanel, EuiResizableButton, { togglePanel }) => {
                collapseFn.current = (id, direction) =>
                  togglePanel(id, { direction });
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
    </GraphContext.Provider>
  );
};

const FlowEditor = () => {
  const diagramEngine = useDiagramEngine();
  return (
    <DiagramEngineContext.Provider value={diagramEngine}>
      <Editor />
    </DiagramEngineContext.Provider>
  );
};

export default FlowEditor;
