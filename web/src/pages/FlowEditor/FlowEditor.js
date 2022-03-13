import React, {
  createContext,
  DragEvent,
  useContext,
  useEffect,
  useReducer,
  useRef,
  useState,
} from "react";
import logo from "../../logo.svg";
import { FaFolderOpen, FaSitemap, FaCubes } from "react-icons/fa";
import {
  EuiHeader,
  EuiPageTemplate,
  EuiHeaderSection,
  EuiHeaderSectionItem,
  EuiIcon,
  EuiHeaderSectionItemButton,
  EuiPanel,
  EuiFlexGroup,
  EuiFlexItem,
  EuiResizableContainer,
  EuiModal,
  EuiModalHeader,
  EuiModalHeaderTitle,
  EuiModalBody,
  EuiForm,
  EuiFilePicker,
  EuiModalFooter,
  EuiButtonEmpty,
  EuiButton,
} from "@elastic/eui";
import ReactFlow, {
  addEdge,
  Background,
  Controls,
  isNode,
  ReactFlowProvider,
  removeElements,
} from "react-flow-renderer";
import { v4 } from "uuid";
import dagre from "dagre";
import "../../dnd.css";
import { getParsedFlow } from "../../api";

const GraphContext = createContext();

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

const onDragStart = (event: DragEvent, nodeType: string) => {
  event.dataTransfer.setData("application/reactflow", nodeType);
  event.dataTransfer.effectAllowed = "move";
};

const NodeGallery = () => {
  return (
    <EuiPanel>
      <EuiFlexGroup direction="column" alignItems="center">
        <EuiFlexItem>
          <div
            className="react-flow__node-input"
            onDragStart={(event: DragEvent) => onDragStart(event, "input")}
            draggable
          >
            Input Node
          </div>
        </EuiFlexItem>
        <EuiFlexItem>
          <div
            className="react-flow__node-default"
            onDragStart={(event: DragEvent) => onDragStart(event, "default")}
            draggable
          >
            Default Node
          </div>
        </EuiFlexItem>
        <EuiFlexItem>
          <div
            className="react-flow__node-output"
            onDragStart={(event: DragEvent) => onDragStart(event, "output")}
            draggable
          >
            Output Node
          </div>
        </EuiFlexItem>
      </EuiFlexGroup>
    </EuiPanel>
  );
};

const nodeWidth = 172;
const nodeHeight = 36;

const dagreGraph = new dagre.graphlib.Graph();
dagreGraph.setDefaultEdgeLabel(() => ({}));

let id = 0;
const getId = () => `dndnode_${id++}`;

const getLayoutedElements = (elements, direction = "TB") => {
  const isHorizontal = Boolean(direction === "LR");
  dagreGraph.setGraph({ rankdir: direction });

  elements.forEach((el) => {
    if (isNode(el)) {
      dagreGraph.setNode(el.id, { width: nodeWidth, height: nodeHeight });
    } else {
      dagreGraph.setEdge(el.source, el.target);
    }
  });

  dagre.layout(dagreGraph);

  return elements.map((el) => {
    if (isNode(el)) {
      const nodeWithPosition = dagreGraph.node(el.id);
      el.targetPosition = isHorizontal ? "left" : "top";
      el.sourcePosition = isHorizontal ? "right" : "bottom";

      // unfortunately we need this little hack to pass a slightly different position
      // to notify react flow about the change. Moreover we are shifting the dagre node position
      // (anchor=center center) to the top left so it matches the react flow node anchor point (top left).
      el.position = {
        x: nodeWithPosition.x - nodeWidth / 2 + Math.random() / 1000,
        y: nodeWithPosition.y - nodeHeight / 2,
      };
    }

    return el;
  });
};

const FlowCanvas = () => {
  const { graph, layoutType } = useContext(GraphContext);
  const [layoutedElements, setLayoutedElements] = useState([]);

  useEffect(() => {
    const layoutedElements = getLayoutedElements(
      graph == null ? [] : graph,
      layoutType
    );
    if (graph !== layoutedElements) {
      setLayoutedElements(layoutedElements);
    }
  }, [graph, layoutType]);

  const onConnect = (params) =>
    setLayoutedElements((els) =>
      addEdge({ ...params, type: "smoothstep", animated: true }, els)
    );

  const onElementsRemove = (elementsToRemove) =>
    setLayoutedElements((els) => removeElements(elementsToRemove, els));

  const reactFlowWrapper = useRef(null);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);

  const onLoad = (_reactFlowInstance) =>
    setReactFlowInstance(_reactFlowInstance);

  const onDragOver = (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  };

  const onDrop = (event) => {
    event.preventDefault();

    const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
    const type = event.dataTransfer.getData("application/reactflow");
    const position = reactFlowInstance.project({
      x: event.clientX - reactFlowBounds.left,
      y: event.clientY - reactFlowBounds.top,
    });
    const newNode = {
      id: getId(),
      type,
      position,
      data: { label: `${type} node` },
    };
    setLayoutedElements((es) => es.concat(newNode));
  };

  return (
    <EuiPanel>
      <EuiFlexGroup>
        <EuiFlexItem className="dndflow">
          <ReactFlowProvider>
            <div className="reactflow-wrapper" ref={reactFlowWrapper}>
              <ReactFlow
                elements={layoutedElements}
                onConnect={onConnect}
                onElementsRemove={onElementsRemove}
                onLoad={onLoad}
                onDrop={onDrop}
                onDragOver={onDragOver}
              >
                <Background variant="dots" gap={15} size={1} />
                <Controls />
              </ReactFlow>
            </div>
          </ReactFlowProvider>
        </EuiFlexItem>
      </EuiFlexGroup>
    </EuiPanel>
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
  const collapseFn = useRef(() => {});
  const [layoutType, toggleLayoutType] = useReducer(
    (t) => (t === "TB" ? "LR" : "TB"),
    "TB"
  );
  const [showOpenFileUi, toggleShowOpenFileUi] = useReducer((t) => !t, false);
  const [graph, setGraph] = useState();
  const [flow, setFlow] = useState();
  return (
    <GraphContext.Provider
      value={{ graph, setGraph, layoutType, flow, setFlow }}
    >
      {showOpenFileUi && <OpenFileUi onClose={toggleShowOpenFileUi} />}
      <EuiPageTemplate pageContentProps={{ paddingSize: "none" }}>
        <Menu
          toggleShowGallery={() => collapseFn.current("panel-gallery", "right")}
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
                      minSize={`${nodeWidth + 30}px`}
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
  );
};
export default FlowEditor;
