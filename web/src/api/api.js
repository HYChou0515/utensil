import axios from "axios";

const instance = axios.create({
  baseURL: "http://127.0.0.1:8000",
  headers: { "Content-Type": "application/json" },
  timeout: 20000,
});

export const restGet = async (url) => {
  return (await instance.get(url)).data;
};

export const restPost = async (url, data) => {
  return (await instance.post(url, data)).data;
};

export const listNodeTasks = async () => {
  return await restGet("/node-tasks");
};

export const getParsedFlow = async (data) => {
  return await restPost("/parse-flow", data);
};
