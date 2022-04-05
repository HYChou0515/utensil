import { createAsyncThunk } from "@reduxjs/toolkit";

import { getNodeTasks } from "../store/actions";
import apiClient from "./api";

export const listNodeTasks = createAsyncThunk(
  getNodeTasks.type,
  apiClient.listNodeTasks
);
