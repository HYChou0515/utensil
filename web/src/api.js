import axios from "axios";

const instance = axios.create({
    baseURL: 'http://127.0.0.1:8000',
    headers: { 'Content-Type': 'application/json' },
    timeout: 20000
});

const restGet = (url) => {
    return instance.get(url);
};

const restPost = (url, data) => {
    return instance.post(url, data);
};

export const listNodeTasks = async () => {
    return restGet('/node-tasks');
};


export const getParsedFlow = async (data) => {
    return restPost('/parse-flow', data);
};
