import {
  EuiButton,
  EuiButtonEmpty,
  EuiFilePicker,
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
  EuiResizableContainer,
} from "@elastic/eui";
import { CanvasWidget } from "@projectstorm/react-canvas-core";
import * as SRD from "@projectstorm/react-diagrams";
import * as _ from "lodash";
import React, {
  createContext,
  useContext,
  useReducer,
  useRef,
  useState,
} from "react";
import { FaCubes, FaFolderOpen, FaSitemap } from "react-icons/fa";
import { v4 } from "uuid";

import { getParsedFlow } from "../../api";
import CanvasWrapper from "../../components/CanvasWrapper";
import GalleryItemWidget from "../../components/GalleryItemWidget";
import { useForceUpdate } from "../../components/hooks";
import logo from "../../logo.svg";

const GraphContext = createContext();
const DiagramEngineContext = createContext();

const Menu = ({
  toggleShowGallery,
  toggleShowAllNodes,
  toggleShowOpenFileUi,
  toggleLayoutType,
}) => {
  const { layoutType } = useContext(GraphContext);
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
          <EuiHeaderSectionItemButton onClick={toggleLayoutType}>
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
      <EuiFlexGroup direction="column" alignItems="center">
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
let count = 0;

const FlowCanvas = () => {
  const { diagramEngine, dagreEngine } = useContext(DiagramEngineContext);
  const onDragOver = (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  };
  const forceUpdate = useForceUpdate();
  const createNode = (name) => new SRD.DefaultNodeModel(name, "rgb(0,192,255)");

  const connectNodes = (nodeFrom, nodeTo, engine) => {
    //just to get id-like structure
    count++;
    const portOut = nodeFrom.addPort(
      new SRD.DefaultPortModel(true, `${nodeFrom.name}-out-${count}`, "Out")
    );
    const portTo = nodeTo.addPort(
      new SRD.DefaultPortModel(false, `${nodeFrom.name}-to-${count}`, "In")
    );
    return portOut.link(portTo);

    // ################# UNCOMMENT THIS LINE FOR PATH FINDING #############################
    // return portOut.link(portTo, engine.getLinkFactories().getFactory(SRD.PathFindingLinkFactory.NAME));
    // #####################################################################################
  };

  const reroute = () => {
    const factory = diagramEngine
      .getLinkFactories()
      .getFactory(SRD.PathFindingLinkFactory.NAME);
    factory.calculateRoutingMatrix();
  };

  const autoDistribute = () => {
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
  //{ id: '7', type: 'output', data: { label: 'output' }, position },
  //{ id: 'e12', source: '1', target: '2', type: edgeType, animated: true },
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

const FlowEditor = () => {
  const diagramEngine = SRD.default();
  const dagreEngine = new SRD.DagreEngine({
    graph: {
      rankdir: "RL",
      ranker: "longest-path",
      marginx: 25,
      marginy: 25,
    },
    includeLinks: true,
  });

  const activeModel = new SRD.DiagramModel();
  diagramEngine.setModel(activeModel);
  const collapseFn = useRef(() => {});
  const [layoutType, toggleLayoutType] = useReducer(
    (t) => (t === "TB" ? "LR" : "TB"),
    "TB"
  );
  const [showOpenFileUi, toggleShowOpenFileUi] = useReducer((t) => !t, false);
  const [graph, setGraph] = useState();
  const [flow, setFlow] = useState();
  return (
    <DiagramEngineContext.Provider value={{ diagramEngine, dagreEngine }}>
      <GraphContext.Provider
        value={{ graph, setGraph, layoutType, flow, setFlow }}
      >
        {showOpenFileUi && <OpenFileUi onClose={toggleShowOpenFileUi} />}
        <EuiPageTemplate pageContentProps={{ paddingSize: "none" }}>
          <Menu
            toggleShowGallery={() =>
              collapseFn.current("panel-gallery", "right")
            }
            toggleShowOpenFileUi={toggleShowOpenFileUi}
            toggleLayoutType={toggleLayoutType}
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
                        <FlowCanvas nodeLayout={layoutType} />
                      </EuiResizablePanel>

                      <EuiResizableButton />

                      <EuiResizablePanel
                        id="panel-gallery"
                        initialSize={20}
                        minSize={`${200}px`}
                      >
                        <NodeGallery />
                      </EuiResizablePanel>
                    </>
                  );
                }}
              </EuiResizableContainer>
            </EuiFlexItem>
          </EuiFlexGroup>
        </EuiPageTemplate>
      </GraphContext.Provider>
    </DiagramEngineContext.Provider>
  );
};
export default FlowEditor;
