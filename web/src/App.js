import React, { useState, Component } from 'react';
import ReactFlow, {
  removeElements,
  addEdge,
  MiniMap,
  Controls,
  Background,
} from 'react-flow-renderer';

import Menubar from "./components/Menubar";
import FlowEditor from "./components/FlowEditor";

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <Menubar/>
        <div style={{ height: 800, width: 1200 }}>
          <FlowEditor />
        </div>
      </header>
    </div>
  );
}

export default App;
