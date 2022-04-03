import { createSlice } from "@reduxjs/toolkit";

import { listNodeTasks } from "../../../api/thunk";
import canvasDomain from "../../../domain/CanvasDomain";

export const flowEditor = createSlice({
  name: "flowEditor",
  initialState: {
    flow: null,
    graph: null,
    collapseRightSideFunc: null,
    collapseTopSideFunc: null,
    forceUpdate: 0,
    isLoading: "",
    isShowOpenFileUi: false,
    isShowGallery: false,
    isShowSettingUi: false,
    usedLayout: "TB",
    nodeTasks: [],
  },
  reducers: {
    createCollapseRightSideFunc: (state, action) => {
      state.collapseRightSideFunc = action.payload;
    },
    toggleShowOpenFileUi: (state) => {
      state.isShowOpenFileUi = !state.isShowOpenFileUi;
    },
    closeOpenFileUi: (state) => {
      state.isShowOpenFileUi = false;
    },
    toggleShowGallery: (state) => {
      state.showGallery = !state.showGallery;
      state.collapseRightSideFunc("panel-gallery", "right");
    },
    toggleShowSettingUi: (state) => {
      state.isShowSettingUi = !state.isShowSettingUi;
    },
    closeSettingUi: (state) => {
      state.isShowSettingUi = false;
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
      canvasDomain.addInPortToSelected(name);
    },
    addOutPortToSelected: (state, action) => {
      const name = action.payload;
      canvasDomain.addOutPortToSelected(name);
    },
    deleteInPortFromSelected: (state, action) => {
      const name = action.payload;
      canvasDomain.deleteInPortFromSelected(name);
    },
    deleteOutPortFromSelected: (state, action) => {
      const name = action.payload;
      canvasDomain.deleteOutPortFromSelected(name);
    },
    autoDistribute: (state, action) => {
      const rankdir = action.payload;
      canvasDomain.autoDistribute(rankdir);
    },
  },
  extraReducers: (builder) => {
    builder.addCase(listNodeTasks.fulfilled, (state, action) => {
      state.nodeTasks = action.payload;
    });
  },
});

// Action creators are generated for each case reducer function
export default flowEditor.reducer;
export const {
  createCollapseRightSideFunc,
  toggleShowOpenFileUi,
  toggleShowGallery,
  toggleUsedLayout,
  closeOpenFileUi,
  closeSettingUi,
  setLoading,
  addInPortToSelected,
  addOutPortToSelected,
  deleteInPortFromSelected,
  autoDistribute,
  deleteOutPortFromSelected,
  toggleShowSettingUi,
} = flowEditor.actions;
