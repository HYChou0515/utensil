import { createAsyncThunk } from "@reduxjs/toolkit";

import { getNodeTasks } from "../store/actions";
import { restGet } from "./api";

export const listNodeTasks = createAsyncThunk(
  getNodeTasks.type,
  async () => await restGet("/node-tasks")
);
