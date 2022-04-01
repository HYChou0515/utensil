import { configureStore } from "@reduxjs/toolkit";
import thunkMiddleware from "redux-thunk";

import canvasReducer from "./features/canvas/flowEditor";

export default configureStore({
  reducer: {
    canvas: canvasReducer,
  },
  middleware: {
    thunkMiddleware,
  },
});
