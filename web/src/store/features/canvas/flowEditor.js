import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";

import { getParsedFlow } from "../../../api/api";
import domain from "../../../domain/CanvasDomain";
import { uploadFlow } from "../../actions";

const getParsedFlowThunk = createAsyncThunk(
  uploadFlow.type,
  async (formData, thunkAPI) => {
    return await getParsedFlow(formData);
  }
);

export const flowEditor = createSlice({
  name: "flowEditor",
  initialState: {
    flow: null,
    graph: null,
    diagramEngine: domain.diagramEngine,
    collapseFn: null,
    forceUpdate: 0,
    isLoading: "",
    isShowOpenFileUi: false,
    isShowGallery: false,
    usedLayout: "TB",
  },
  reducers: {
    createCollapseFn: (state, action) => {
      state.collapseFn = action.payload;
    },
    toggleShowOpenFileUi: (state) => {
      state.isShowOpenFileUi = !state.isShowOpenFileUi;
    },
    closeOpenFileUi: (state) => {
      state.isShowOpenFileUi = false;
    },
    toggleShowGallery: (state) => {
      state.showGallery = !state.showGallery;
      state.collapseFn("panel-gallery", "right");
    },
    toggleUsedLayout: (state) => {
      let newLayout;
      if (state.usedLayout === "TB") newLayout = "LR";
      else if (state.usedLayout === "LR") newLayout = "TB";
      state.usedLayout = newLayout;
    },
    setLoading: (state, action) => {
      state.isLoading = action.payload;
    },
    addInPortToSelected: (state, action) => {
      const name = action.payload;
      domain.addInPortToSelected(name);
    },
    addOutPortToSelected: (state, action) => {
      const name = action.payload;
      domain.addOutPortToSelected(name);
    },
    deleteInPortFromSelected: (state, action) => {
      const name = action.payload;
      domain.deleteInPortFromSelected(name);
    },
    deleteOutPortFromSelected: (state, action) => {
      const name = action.payload;
      domain.deleteOutPortFromSelected(name);
    },
    autoDistribute: (state, action) => {
      const rankdir = action.payload;
      domain.autoDistribute(rankdir);
    },
  },
  extraReducers: (builder) => {
    builder.addCase(getParsedFlowThunk.fulfilled, (state, action) => {
      state.flow = action.payload;
      state.graph = domain.parseFlowToGraph(action.payload);
      state.isLoading = "";
      state.isShowOpenFileUi = false;
    });
  },
});

// Action creators are generated for each case reducer function
export default flowEditor.reducer;
export const {
  createCollapseFn,
  toggleShowOpenFileUi,
  toggleShowGallery,
  toggleUsedLayout,
  closeOpenFileUi,
  setLoading,
  addInPortToSelected,
  addOutPortToSelected,
  deleteInPortFromSelected,
  autoDistribute,
  deleteOutPortFromSelected,
} = flowEditor.actions;
