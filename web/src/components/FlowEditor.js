import React, { useState, useRef, DragEvent } from 'react';
import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  removeElements,
  Controls,
  Background,
} from 'react-flow-renderer';

import '../dnd.css';
import {
  Transition,
  Menu,
  Icon,
  Grid,
  Container,
  List,
  Segment,
} from 'semantic-ui-react'
import ConditionNode from "./ConditionNode";
import {list_node_tasks} from "../api"


const FlowMenu = ({
  toggleShowGallery,
  toggleShowAllNodes,
}) => {
  const [activeItem, setActivaItem] = useState(null);
  const toggleActiveItem = (selectedItem) => setActivaItem(activeItem !== selectedItem ? selectedItem: '');

  const onMenuItemCheck = (e, {name}) => {
    toggleActiveItem(name);
    if (name === 'show-gallery')
      toggleShowGallery();
    if (name === 'help') {
      list_node_tasks().then((o) => console.log(o));
    }
    if (name === 'show-all-nodes') {
      toggleShowAllNodes();
    }
  }

  return (
    <Menu icon>
      <Menu.Item
        name='new-flow'
        active={activeItem === 'new-flow'}
        onClick={onMenuItemCheck}
      >
        <Icon name='file' />
      </Menu.Item>

      <Menu.Item
        name='open-flow'
        active={activeItem === 'open-flow'}
        onClick={onMenuItemCheck}
      >
        <Icon name='folder open' />
      </Menu.Item>

      <Menu.Item
        name='save-flow'
        active={activeItem === 'save-flow'}
        onClick={onMenuItemCheck}
      >
        <Icon name='save' />
      </Menu.Item>

      <Menu.Item
        name='clone-flow'
        active={activeItem === 'clone-flow'}
        onClick={onMenuItemCheck}
      >
        <Icon name='clone' />
      </Menu.Item>

      <Menu.Menu position='right'>

        <Menu.Item
          name='show-all-nodes'
          active={activeItem === 'show-all-nodes'}
          onClick={onMenuItemCheck}
        >
          <Icon name='list layout' />
        </Menu.Item>

        <Menu.Item
          name='show-gallery'
          active={activeItem === 'show-gallery'}
          onClick={onMenuItemCheck}
        >
          <Icon name='sitemap' />
        </Menu.Item>

        <Menu.Item
          name='help'
          active={activeItem === 'help'}
          onClick={onMenuItemCheck}
        >
          Help
        </Menu.Item>
      </Menu.Menu>
    </Menu>
  )
};

const onDragStart = (event: DragEvent, nodeType: string) => {
  event.dataTransfer.setData('application/reactflow', nodeType);
  event.dataTransfer.effectAllowed = 'move';
};

const NodeGallery = ({isShow, zIndex}) => {
  return (
    <Transition.Group animation='fly down' duration={500}>
      {isShow && (
        <div style={{zIndex:{zIndex}, position: 'absolute'}}>
          <Segment raised style={{width: '180px'}}>
            <List>
              <List.Item>
                <div className="react-flow__node-input"
                   onDragStart={(event: DragEvent) => onDragStart(event, 'input')}
                   draggable
                >
                  Input Node
                </div>
              </List.Item>
              <List.Item>
                <div className="react-flow__node-default"
                   onDragStart={(event: DragEvent) => onDragStart(event, 'default')}
                   draggable
                >
                  Default Node
                </div>
              </List.Item>
              <List.Item>
                <div className="react-flow__node-condition"
                   onDragStart={(event: DragEvent) => onDragStart(event, 'condition')}
                   draggable
                >
                  Condition Node
                </div>
              </List.Item>
              <List.Item>
                <div className="react-flow__node-output"
                   onDragStart={(event: DragEvent) => onDragStart(event, 'output')}
                   draggable
                >
                  Output Node
                </div>
              </List.Item>
            </List>
          </Segment>
        </div>
      )}
    </Transition.Group>
  )
};


const initialElements = [
  {
    id: '1',
    type: 'input',
    data: { label: 'input node' },
    position: { x: 250, y: 5 },
  },
];

let id = 0;
const getId = () => `dndnode_${id++}`;

const nodeTypes = {
  condition: ConditionNode,
};

const FlowCanvas = () => {
  const reactFlowWrapper = useRef(null);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const [elements, setElements] = useState(initialElements);
  const onConnect = (params) => setElements((els) => addEdge(params, els));
  const onElementsRemove = (elementsToRemove) =>
    setElements((els) => removeElements(elementsToRemove, els));

  const onLoad = (_reactFlowInstance) =>
    setReactFlowInstance(_reactFlowInstance);

  const onDragOver = (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  };

  const onDrop = (event) => {
    event.preventDefault();

    const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
    const type = event.dataTransfer.getData('application/reactflow');
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

    setElements((es) => es.concat(newNode));
  };

  return (
    <Segment className="dndflow">
      <ReactFlowProvider>
        <div className="reactflow-wrapper" ref={reactFlowWrapper}>
          <ReactFlow
            elements={elements}
            onConnect={onConnect}
            onElementsRemove={onElementsRemove}
            onLoad={onLoad}
            onDrop={onDrop}
            onDragOver={onDragOver}
            nodeTypes={nodeTypes}
          >
            <Background
              variant="dots"
              gap={15}
              size={1}
            />
            <Controls />
          </ReactFlow>
        </div>
      </ReactFlowProvider>
    </Segment>
  );
};

const FlowEditor = () => {
  const [siedColumn, setSideColumn] = useState('')
  const toggleSideColumn = (selectedColumn) => setSideColumn(siedColumn !== selectedColumn ? selectedColumn: '');
  const toggleShowGallery = () => toggleSideColumn('node-gallery');
  const toggleShowAllNodes = () => toggleSideColumn('all-nodes');

  return (
    <Container>
      <FlowMenu
        toggleShowGallery={toggleShowGallery}
        toggleShowAllNodes={toggleShowAllNodes}
      />
      <Grid stackable columns='equal'>
        <Grid.Column width={13}>
          <FlowCanvas/>
        </Grid.Column>
        <Grid.Column width={1}>
          <NodeGallery zindex={1} isShow={siedColumn==='node-gallery'}/>
          <NodeGallery zindex={2} isShow={siedColumn==='all-nodes'}/>
        </Grid.Column>
      </Grid>
    </Container>
  );
};

export default FlowEditor;
