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
import { v4 } from "uuid";

import { getParsedFlow } from "../../api";
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
import logo from "../../logo.svg";

const GraphContext = createContext();
const DiagramEngineContext = createContext();
const DiagramControlContext = createContext();

const Menu = ({
  toggleShowGallery,
  toggleShowAllNodes,
  toggleShowOpenFileUi,
}) => {
  const { eventQueue, setNewEventComing } = useContext(DiagramControlContext);
  const layoutTypeSwitch = (t) => (t === "TB" ? "LR" : "TB");
  const [layoutType, toggleLayoutType] = useReducer(layoutTypeSwitch, "TB");
  const onToggleShowGallery = () => {
    eventQueue.push([
      "autoDistribute",
      { rankdir: layoutTypeSwitch(layoutType) },
    ]);
    setNewEventComing();
    toggleLayoutType();
  };
  return (
    <EuiHeader>
      <EuiHeaderSection grow={false}>
        <EuiHeaderSectionItem>
          <EuiHeaderSectionItemButton onClick={() => console.log("ho")}>
            <EuiIcon type={logo} size="l" />
          </EuiHeaderSectionItemButton>
        </EuiHeaderSectionItem>
        <EuiHeaderSectionItem>
          <EuiHeaderSectionItemButton onClick={toggleShowOpenFileUi}>
            <EuiIcon type={FaFolderOpen} />
          </EuiHeaderSectionItemButton>
        </EuiHeaderSectionItem>
      </EuiHeaderSection>

      <EuiHeaderSection side="right">
        <EuiHeaderSectionItem>
          <EuiHeaderSectionItemButton onClick={toggleShowGallery}>
            <EuiIcon type={FaCubes} />
          </EuiHeaderSectionItemButton>
        </EuiHeaderSectionItem>
        <EuiHeaderSectionItem>
          <EuiHeaderSectionItemButton onClick={onToggleShowGallery}>
            <EuiIcon
              type={FaSitemap}
              style={{
                transform: `rotate(${layoutType === "TB" ? 0 : 270}deg)`,
              }}
            />
          </EuiHeaderSectionItemButton>
        </EuiHeaderSectionItem>
      </EuiHeaderSection>
    </EuiHeader>
  );
};

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

const parseFlowToGraph = (flow) => {
  const els = [];
  if (flow?.nodes == null) return;
  flow?.nodes.forEach((node) => {
    els.push({
      id: node.name,
      type: node.end_of_flow ? "output" : node.switchon ? "input" : "default",
      data: { label: node.name },
      position: {
        x: Math.random() * window.innerWidth - 100,
        y: Math.random() * window.innerHeight,
      },
    });
    node.receivers.forEach((recv) => {
      els.push({
        id: `${node.name}-send-${recv}`,
        source: node.name,
        target: recv,
        type: "smoothstep",
        animated: true,
      });
    });
    node.callees.forEach((_callee) => {
      const callee = _callee === ":self:" ? node.name : _callee;
      if (callee !== "SWITCHON") {
        els.push({
          id: `${node.name}-call-${callee}`,
          source: node.name,
          target: callee,
          type: "smoothstep",
          animated: true,
        });
      }
    });
  });
  return els;
};

const OpenFileUi = ({ onClose }) => {
  const { setGraph, setFlow } = useContext(GraphContext);
  const formId = v4();
  const [file, setFile] = useState();
  const [isLoading, setIsLoading] = useState(false);
  const [isInvalidFile, setIsInvalidFile] = useState(false);
  const onOpenClick = () => {
    if (file) {
      setIsInvalidFile(false);
      setIsLoading(true);
      let formData = new FormData();
      formData.append("file", file, file.name);
      getParsedFlow(formData).then((newFlow) => {
        setFlow(newFlow);
        setGraph(parseFlowToGraph(newFlow));
        setIsLoading(false);
        onClose();
      });
    } else {
      setIsInvalidFile(true);
    }
  };
  return (
    <EuiModal onClose={onClose}>
      <EuiModalHeader>
        <EuiModalHeaderTitle>
          <h1>Drop a file here or click to select a file</h1>
        </EuiModalHeaderTitle>
      </EuiModalHeader>

      <EuiModalBody>
        <EuiForm id={formId}>
          <EuiFilePicker
            onChange={(f) => setFile(f.item(0))}
            isLoading={isLoading}
            isInvalid={isInvalidFile}
          />
        </EuiForm>
      </EuiModalBody>

      <EuiModalFooter>
        <EuiButtonEmpty onClick={onClose}>Cancel</EuiButtonEmpty>
        <EuiButton
          type="submit"
          form={formId}
          fill
          onClick={onOpenClick}
          isDisabled={(file?.size ?? 0) === 0}
          isLoading={isLoading}
        >
          Open
        </EuiButton>
      </EuiModalFooter>
    </EuiModal>
  );
};

const Editor = () => {
  const collapseFn = useRef(() => {});
  const [showOpenFileUi, toggleShowOpenFileUi] = useReducer((t) => !t, false);
  const [graph, setGraph] = useState();
  const [flow, setFlow] = useState();
  const [eventQueue] = useState([]);
  const [newEventComing, setNewEventComing] = useState(0);
  return (
    <GraphContext.Provider value={{ graph, setGraph, flow, setFlow }}>
      <DiagramControlContext.Provider
        value={{ eventQueue, newEventComing, setNewEventComing }}
      >
        {showOpenFileUi && <OpenFileUi onClose={toggleShowOpenFileUi} />}
        <EuiPageTemplate pageContentProps={{ paddingSize: "none" }}>
          <Menu
            toggleShowGallery={() =>
              collapseFn.current("panel-gallery", "right")
            }
            toggleShowOpenFileUi={toggleShowOpenFileUi}
          />
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
      </DiagramControlContext.Provider>
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
