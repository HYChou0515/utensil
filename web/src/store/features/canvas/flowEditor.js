import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";

import { getParsedFlow } from "../../../api/api";
import { parseFlowToGraph } from "../../../domain/domain";
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
    isLoading: "",
    isShowOpenFileUi: false,
    isShowGallery: false,
    usedLayout: "TB",
  },
  reducers: {
    toggleShowOpenFileUi: (state) => {
      state.isShowOpenFileUi = !state.isShowOpenFileUi;
    },
    closeOpenFileUi: (state) => {
      state.isShowOpenFileUi = false;
    },
    toggleShowGallery: (state) => {
      state.showGallery = !state.showGallery;
    },
    toggleUsedLayout: (state) => {
      let newLayout;
      if (state.usedLayout === "TB") newLayout = "LR";
      else if (state.usedLayout === "LR") newLayout = "TB";
      console.log(newLayout);
      state.usedLayout = newLayout;
    },
    setLoading: (state, action) => {
      state.isLoading = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(getParsedFlowThunk.fulfilled, (state, action) => {
      state.flow = action.payload;
      state.graph = parseFlowToGraph(action.payload);
      state.isLoading = "";
      state.isShowOpenFileUi = false;
    });
  },
});

// Action creators are generated for each case reducer function
export const {
  toggleShowOpenFileUi,
  toggleShowGallery,
  toggleUsedLayout,
  closeOpenFileUi,
  setLoading,
} = flowEditor.actions;
export default flowEditor.reducer;
