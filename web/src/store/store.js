import { configureStore } from "@reduxjs/toolkit";

import flowEditorReducer from "./features/canvas/flowEditor";

export default configureStore({
  reducer: {
    flowEditor: flowEditorReducer,
  },
});
