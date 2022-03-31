import { configureStore } from "@reduxjs/toolkit";

import canvasReducer from "./features/canvas/flowEditor";

export default configureStore({
  reducer: {
    canvas: canvasReducer,
  },
});
