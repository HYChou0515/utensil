import { createSlice } from "@reduxjs/toolkit";

export const flowEditor = createSlice({
  name: "flowEditor",
  initialState: {
    value: 0,
    isShowOpenFileUi: false,
    isShowGallery: false,
    usedLayout: "TB",
  },
  reducers: {
    toggleShowOpenFileUi: (state) => {
      state.isShowOpenFileUi = !state.isShowOpenFileUi;
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
    increment: (state) => {
      // Redux Toolkit allows us to write "mutating" logic in reducers. It
      // doesn't actually mutate the state because it uses the Immer library,
      // which detects changes to a "draft state" and produces a brand new
      // immutable state based off those changes
      state.value += 1;
    },
    decrement: (state) => {
      state.value -= 1;
    },
    incrementByAmount: (state, action) => {
      state.value += action.payload;
    },
  },
});

// Action creators are generated for each case reducer function
export const {
  toggleShowOpenFileUi,
  toggleShowGallery,
  toggleUsedLayout,
  increment,
  decrement,
  incrementByAmount,
} = flowEditor.actions;
export default flowEditor.reducer;
