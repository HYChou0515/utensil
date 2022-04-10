import axios from "axios";

const instance = axios.create({
  baseURL: "http://127.0.0.1:8000",
  headers: { "Content-Type": "application/json" },
  timeout: 20000,
});

const restGet = async (url) => {
  return (await instance.get(url)).data;
};

const restPost = async (url, data) => {
  return (await instance.post(url, data)).data;
};

class ApiClient {
  listNodeTasks = async () => {
    return await restGet("/node-tasks");
  };

  getSourceCode = async (module, taskName) => {
    return await restGet(`/task-source-code/${module}/${taskName}`);
  };

  getParsedFlow = async (data) => {
    return await restPost("/parse-flow", data);
  };

  postGraph = async (data) => {
    return await restPost("/graph", data);
  };
}

export default new ApiClient();
